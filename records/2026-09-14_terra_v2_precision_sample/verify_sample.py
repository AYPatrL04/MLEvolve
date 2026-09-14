"""Bounded CPU checks of the untouched Terra draft; never invokes main()."""
import ast
import importlib.util
import json
import time
from pathlib import Path

import torch

from agents.hardware_context import get_hardware_context_for_stage
from agents.precision_validation import validate_training_precision
from engine.preflight import ModelPreflightGate, inspect_adapter
from engine.search_node import SearchNode

OUT = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.set_num_threads(2)
    harness = load_module('terra_sample_harness', OUT / 'audit_sample.py')
    agent, _, _, _ = harness.build_agent()
    code = (OUT / 'draft.py').read_text()
    tree = ast.parse(code)
    node = SearchNode(code=code, plan=(OUT / 'plan.txt').read_text(), stage='draft')
    context = get_hardware_context_for_stage(agent, 'draft', code=code)
    issues = validate_training_precision(agent, code, context=context)
    started = time.time()
    outcome = ModelPreflightGate(agent.cfg).run(node, generated=True, attempt=0)
    finished = time.time()
    harness.write_json(OUT / 'preflight.json', outcome.to_dict())
    report = {
        'preflight_started_unix': started, 'preflight_finished_unix': finished,
        'syntax_valid': True, 'adapter_complete': inspect_adapter(code).complete,
        'precision_issues': [issue.to_dict() for issue in issues],
        'preflight_admitted': outcome.admitted, 'preflight_status': outcome.status,
        'whole_model_half_cast_lines': [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                                       and isinstance(n.func, ast.Attribute) and n.func.attr in {'half', 'bfloat16'}],
        'training_entrypoint_executed': False, 'gpu_executed': False,
    }
    dynamic_start = time.time()
    try:
        module = load_module('untouched_terra_draft', OUT / 'draft.py')
        adapter = module.CandidateAdapter()
        model = adapter.build_model({}).to('cpu')
        optimizer = adapter.build_optimizer(model, {})
        scenario = {'batch_size': 2, 'fixture': {'image': [3, 256, 256], 'tabular': [12]}}
        batch = adapter.build_train_batch(scenario, torch.device('cpu'))
        model.train()
        loss = adapter.training_step(model, batch, {})
        loss.backward()
        report['parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
        report['gradient_dtypes'] = sorted({str(p.grad.dtype) for p in model.parameters() if p.grad is not None})
        report['all_gradients_finite'] = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        report['parameter_count'] = sum(p.numel() for p in model.parameters())
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.step()
        report['optimizer_updates_parameters'] = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        report['floating_optimizer_state_dtypes'] = sorted({str(v.dtype) for state in optimizer.state.values()
                                                         for v in state.values() if torch.is_tensor(v) and v.is_floating_point()})
        report['cpu_training_loss_dtype'] = str(loss.dtype)
        report['cpu_training_loss_finite'] = bool(torch.isfinite(loss))
        model.eval()
        with torch.no_grad():
            val_loss = adapter.validation_step(model, adapter.build_validation_batch(scenario, torch.device('cpu')), {})
        report['cpu_validation_loss_dtype'] = str(val_loss.dtype)
        report['cpu_validation_loss_finite'] = bool(torch.isfinite(val_loss))
        try:
            module.mean_squared_error([1.0, 2.0], [1.0, 2.0], squared=False)
            report['installed_rmse_api_compatible'] = True
        except TypeError as exc:
            report['installed_rmse_api_compatible'] = False
            report['rmse_api_error'] = str(exc)
        import sklearn
        report['host_versions'] = {'torch': torch.__version__, 'sklearn': sklearn.__version__}
    except Exception as exc:
        report['cpu_smoke_error'] = f'{type(exc).__name__}: {exc}'
    report['cpu_smoke_started_unix'] = dynamic_start
    report['cpu_smoke_finished_unix'] = time.time()
    harness.write_json(OUT / 'checks.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
