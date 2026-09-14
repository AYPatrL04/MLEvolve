"""Current-code graph/prompt replay and bounded checks of a corrected sample."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import torch

from agents import draft_agent
from agents.coder import base_coder
from agents.hardware_context import get_hardware_context_for_stage
from agents.precision_validation import validate_training_precision
from agents.training_contract_validation import validate_training_contract
from engine.preflight import ModelPreflightGate
from engine.search_node import SearchNode
from knowledge.records import validate_record
from utils.training_diagnostics import TrainingDiagnostics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    torch.set_num_threads(2)
    harness = load('fresh_harness', OUT.parent / 'audit_sample.py')
    harness.OUT = OUT
    agent, graph, snapshot, ingestion = harness.build_agent()
    graph_records = [record for row in graph.get_feature_neighborhood(hardware_terms=['NVIDIA Tesla V100 32GB'])['features']
                     for record in row.get('design_records_v2', [])]
    for record in graph_records:
        validate_record(record)
    captured = []
    original_fit = draft_agent.fit_prompt

    def fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        captured.extend(result[1])
        write('context_sizing.json', result[2])
        return result

    def capture(**kwargs):
        write('prompt.json', kwargs['prompt'])
        raise harness.CaptureComplete()

    with patch.object(draft_agent, 'fit_prompt', fit), patch.object(base_coder, 'generate', capture):
        try:
            draft_agent.run(agent)
        except harness.CaptureComplete:
            pass
    write('knowledge_records.json', captured)
    write('source_snapshot.json', snapshot)
    write('graph_records.json', graph_records)
    prompt = (OUT / 'prompt.json').read_text()
    assert any('hardware_knowledge:HAS_FEATURE:' in ref for record in captured for ref in record['evidence_refs'])
    assert 'select_validated_precision' in prompt and 'model.half()' in prompt
    assert 'np.sqrt(mean_squared_error' in prompt
    code = (OUT / 'draft.py').read_text()
    ast.parse(code)
    node = SearchNode(code=code, plan='Locally corrected original Terra sample; no new LLM call.', stage='draft')
    context = get_hardware_context_for_stage(agent, 'draft', code=code)
    assert not validate_training_precision(agent, code, context=context)
    assert not validate_training_contract(code, scheduler_enabled=False)
    started = time.time()
    outcome = ModelPreflightGate(agent.cfg).run(node, generated=True, attempt=0)
    preflight_finished = time.time()
    write('preflight.json', outcome.to_dict())
    assert outcome.admitted, outcome.to_dict()
    write('preflight_report.json', json.loads(Path(outcome.report_path).read_text()))
    script = load('corrected_sample', OUT / 'draft.py')
    # A controlled CPU autocast failure must restore state and make one finite FP32 update.
    model = script.ImageTabularRegressor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    image, tabular, target = torch.zeros(2, 3, 256, 256), torch.zeros(2, 12), torch.zeros(2)
    reference = deepcopy(model)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001)
    rng = torch.get_rng_state()
    def unstable_loss(prediction, expected):
        return prediction.sum() * float('nan') if torch.is_autocast_enabled('cpu') else torch.nn.functional.mse_loss(prediction, expected)
    with TrainingDiagnostics(model, optimizer) as diagnostics:
        loss, enabled, scaler = script.finite_training_step(model, optimizer, unstable_loss, image, tabular, target,
                                                           torch.device('cpu'), True, torch.amp.GradScaler('cpu'), diagnostics)
        assert not enabled and torch.isfinite(loss) and diagnostics.completed_updates == 1
    torch.set_rng_state(rng)
    torch.nn.functional.mse_loss(reference(image, tabular), target).backward()
    ref_optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), reference.parameters()))
    write('numerical_fallback.json', {'cpu_fault_injection': True, 'restored_rng_and_state': True,
                                     'matches_finite_fp32_reference_update': True, 'gpu_test': False})
    fallback_finished = time.time()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONPATH=str(ROOT / 'nn-model-preflight-checker/src') + ':' + str(ROOT),
               MLEVOLVE_INPUT_DIR=str(OUT / 'workspace/input'))
    process = subprocess.run([str(ROOT / '.venv/bin/python'), str(OUT / 'draft.py')], cwd=OUT / 'workspace',
                             env=env, capture_output=True, text=True, timeout=60)
    (OUT / 'execution.stdout.txt').write_text(process.stdout)
    (OUT / 'execution.stderr.txt').write_text(process.stderr)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip().splitlines()[-1].startswith('Final Validation Score: ')
    import pandas as pd
    submission = pd.read_csv(OUT / 'workspace/submission/submission.csv')
    assert list(submission) == ['Id', 'Pawpularity'] and len(submission) == 4
    assert submission.Pawpularity.between(0, 100).all()
    checkpoint = torch.load(OUT / 'workspace/working/petfinder_checkpoint.pt', weights_only=False, map_location='cpu')
    score = float(process.stdout.strip().splitlines()[-1].split(': ')[1])
    epoch_scores = [float(line.split('val_rmse=')[1]) for line in process.stdout.splitlines() if line.startswith('Epoch ')]
    assert abs(score - min(epoch_scores)) < 1e-4
    assert checkpoint['precision_quality']['precision'] == 'fp32'
    write('result.json', {'graph_projected_records_valid': len(graph_records), 'prompt_records': len(captured),
                         'graph_evidence_in_prompt': True, 'new_llm_calls': 0, 'candidate_origin': 'local correction of archived Terra output',
                         'preflight_status': outcome.status, 'preflight_admitted': outcome.admitted,
                         'preflight_started': started, 'preflight_finished': preflight_finished,
                         'fallback_finished': fallback_finished, 'execution_finished': time.time(),
                         'numerical_fallback_matches_fp32': True, 'submission_rows': len(submission),
                         'synthetic_rmse': score, 'best_epoch': checkpoint['epoch'],
                         'selected_precision': checkpoint['precision_quality']['precision'],
                         'quality_status': checkpoint['precision_quality']['status'],
                         'gpu_training': False, 'task_quality_validated': False})
    print((OUT / 'result.json').read_text())


if __name__ == '__main__':
    main()
