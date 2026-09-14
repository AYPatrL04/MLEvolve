"""Fresh current-corpus sample; isolates storage and transports one Terra response."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

from agents import draft_agent
from agents.coder import base_coder
from benchmarks.petfinder_workflow_matrix import COLUMNS, make_agent, prepare_data, write_json
from hardware_knowledge_graph.client import HardwareKnowledgeClient
from hardware_knowledge_graph.config import HardwareKnowledgeSettings
from knowledge.runtime import pin_version
from localml_scheduler.hardware_knowledge.records import load_hardware_knowledge_from_schema
from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


class SnapshotGraph(HardwareKnowledgeGraphStore):
    """In-memory storage I/O; uses production ingestion, ranking and projection."""
    def __init__(self, settings):
        super().__init__(settings)
        self.hardware = {}
        self.features = {}
        self.relationships = {}
        self.reads = []

    def _run_write(self, cypher, params=None):
        params = params or {}
        if 'MERGE (h:HardwareSpec' in cypher:
            self.hardware[params['hardware_id']] = params['props']
        elif 'MERGE (f:Feature' in cypher:
            self.features[params['feature_id']] = params['props']
        elif 'MERGE (h)-[r:HAS_FEATURE]' in cypher:
            self.relationships[(params['hardware_id'], params['feature_id'])] = params['props']
        elif not cypher.startswith('CREATE '):
            raise AssertionError('Unexpected graph mutation')
        return []

    def _query_neighborhood_rows(self, *, hardware_terms, row_limit, feature_ids=None):
        self.reads.append({'hardware_terms': hardware_terms, 'row_limit': row_limit})
        rows = []
        for (hardware_id, feature_id), relationship in self.relationships.items():
            hardware = self.hardware[hardware_id]
            names = [hardware_id, hardware['name'], hardware['name_key'], *hardware.get('aliases', [])]
            if not any(str(term).lower() in str(name).lower() for term in hardware_terms for name in names):
                continue
            if feature_ids and feature_id not in feature_ids:
                continue
            rows.append({'hardware': hardware, 'feature': self.features[feature_id], 'relationship': relationship})
        return rows[:row_limit]


class CaptureComplete(BaseException):
    pass


def build_agent():
    workspace = OUT / 'workspace'
    workspace.mkdir(exist_ok=True)
    if not (workspace / 'input/train.csv').exists():
        prepare_data(workspace)
    agent = make_agent(workspace, 'normal', True)
    agent.acfg.code.model = 'gpt-5.6-terra'
    agent.design_knowledge_version = pin_version(workspace, 'v2', resuming=False)
    profile = OUT / 'v100_32gb.yaml'
    profile.write_text('schema_version: 1\nvendor: nvidia\nname: Tesla V100 32GB\narchitecture: volta\ncompute_capability: "7.0"\nvram_bytes: 34359738368\nnative_training_dtypes: [fp32, fp16]\nunsupported_features: []\n')
    agent.cfg.preflight.target_profile = str(profile)
    agent.task_desc = (
        f'Build a complete PetFinder-style image plus tabular regression script. Input root: {workspace / "input"}. '
        'train.csv has Id, twelve binary metadata columns, and Pawpularity in [0,100]; test.csv omits the target. '
        f'Metadata columns: {COLUMNS}. JPEG paths are input/train/<Id>.jpg and input/test/<Id>.jpg. '
        'Choose a suitable compact CNN architecture with a tabular branch. Preserve image [3,256,256] and tabular [12] interfaces. '
        'No pretrained checkpoints are supplied and no network downloads are available; train from scratch. '
        'Use one deterministic train/validation split and at most two epochs. Fit preprocessing only on training data. '
        'Report RMSE in original target units and produce Id,Pawpularity predictions in [0,100] under the submission directory. '
        'The objective is to shorten training time per epoch while preserving task quality as the first prerequisite. '
        'Target deployment is one NVIDIA Tesla V100 32GB GPU with PyTorch 2.5.1 and CUDA 12.4, normal precision policy. '
        'These target software versions are sample assumptions, not observations of this CPU host. '
        'The supplied 24 training and 4 test samples are synthetic interface fixtures; do not claim measured accuracy or speed benefits. '
        'Implement real CPU-safe CandidateAdapter methods that exercise the same model, inputs, loss, and optimizer; '
        'all training and submission side effects belong inside the main guard.'
    )
    settings = HardwareKnowledgeSettings(runtime_root=OUT / 'isolated_hwdb', code_knowledge={'enabled': False})
    client = HardwareKnowledgeClient(settings, include_profile_evidence=False)
    client._probe_status = {
        'ok': True, 'source': 'explicit_sample_target_not_live_probe',
        'hardware_profile': {'hardware_key': 'terra-sample-v100', 'os_name': 'linux',
                             'gpu_name': 'NVIDIA Tesla V100 32GB', 'total_vram_mb': 32768,
                             'compute_capability': '7.0', 'cuda_runtime': '12.4', 'torch_version': '2.5.1'},
    }
    bundle = load_hardware_knowledge_from_schema(ROOT / 'schema')
    hardware = [h for h in bundle['hardware'] if h['name'] == 'NVIDIA Tesla V100 32GB']
    assert len(hardware) == 1
    relationships = [r for r in bundle['relationships'] if r['hardware_id'] == hardware[0]['hardware_id']]
    feature_ids = {r['feature_id'] for r in relationships}
    snapshot = {'hardware': hardware, 'features': [f for f in bundle['features'] if f['feature_id'] in feature_ids], 'relationships': relationships}
    graph = SnapshotGraph(settings)
    ingestion = graph.ingest_bundle(snapshot)
    client._hardware_knowledge_store = graph
    agent.hardware_knowledge_client = client
    return agent, graph, snapshot, ingestion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'finish'])
    args = parser.parse_args()
    agent, graph, snapshot, ingestion = build_agent()
    calls = []
    selected = []
    original_fit = draft_agent.fit_prompt
    started = time.time()

    def capture_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        selected.extend(result[1])
        write_json(OUT / 'context_sizing.json', result[2])
        return result

    def transport(**kwargs):
        prompt = kwargs['prompt']
        if args.action == 'prepare':
            write_json(OUT / 'prompt.json', prompt)
            (OUT / 'prompt.txt').write_text('\n\n'.join(f'[{role}]\n{text}' for role, text in prompt.items()))
            raise CaptureComplete()
        assert prompt == json.loads((OUT / 'prompt.json').read_text()), 'Prompt changed after generation'
        calls.append({'role': 'draft', 'transport': 'Terra subagent saved response', 'model': 'gpt-5.6-terra'})
        assert len(calls) == 1, 'Sample response required an extraction retry'
        return (OUT / 'response.txt').read_text()

    try:
        with patch.object(draft_agent, 'fit_prompt', capture_fit), patch.object(base_coder, 'generate', transport):
            node = draft_agent.run(agent)
    except CaptureComplete:
        node = None
    write_json(OUT / 'knowledge_records.json', selected)
    write_json(OUT / 'source_snapshot.json', snapshot)
    write_json(OUT / 'ingestion.json', ingestion)
    write_json(OUT / 'graph_reads.json', graph.reads)
    if args.action == 'prepare':
        write_json(OUT / 'manifest.json', {
            'prepared_unix': started, 'knowledge_version': 'v2', 'precision_mode': 'normal',
            'model': 'gpt-5.6-terra', 'hardware': 'NVIDIA Tesla V100 32GB',
            'source': 'schema/hardware_knowledge_graph.json',
            'source_sha256': hashlib.sha256((ROOT / 'schema/hardware_knowledge_graph.json').read_bytes()).hexdigest(),
            'knowledge_record_count': len(selected), 'prompt_chars': len((OUT / 'prompt.txt').read_text()),
            'storage': 'isolated in-memory graph I/O, production ingestion/client/filter/projection/prompt assembly',
            'historical_runs_read': False, 'lesson_profiles_enabled': False, 'qdrant_code_index_enabled': False,
            'scheduler_enabled': False, 'training_entrypoint_executed': False,
        })
    else:
        assert node and node.generation_strategy == 'single_pass'
        (OUT / 'draft.py').write_text(node.code)
        (OUT / 'plan.txt').write_text(node.plan)
        write_json(OUT / 'node.json', {'id': node.id, 'generation_strategy': node.generation_strategy,
                                      'pipeline_decision': node.pipeline_decision, 'stage_note_board': node.stage_note_board,
                                      'diagnostics': node.diagnostics, 'calls': calls})
    print(json.dumps({'action': args.action, 'records': len(selected), 'graph_reads': len(graph.reads), 'draft_calls': len(calls)}))


if __name__ == '__main__':
    main()
