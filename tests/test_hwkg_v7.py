import copy
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from hwkg.adaptive_policy import guidance_v7, parent_feedback
from hwkg.injection import hardware_guidance
from tests.test_hwkg_v6 import GRAPH, FEATURES, DEVICES, IDS, RUNTIME

CODE = 'import torch\nm = model.to(memory_format=torch.channels_last)\nwith torch.autocast("cuda", dtype=torch.bfloat16):\n    pass'


def receipt(code=CODE, **changes):
    row = {'schema': 'shared-runtime-v2', 'origin': 'shared-parent-execution-adapter',
           'code_sha256': hashlib.sha256(code.encode()).hexdigest(), 'execution_succeeded': True,
           'exception': None, 'first_step_finite': True,
           'input_contract': {'status': 'observed_pass', 'violations': []},
           'gpu': {'name': DEVICES['5080'].name, 'capability': [12, 0]},
           'peak_allocated_bytes': 2 * 1024**3, 'window_validity': 'single_optimizer_observed',
           'training_observations': [{'autocast_dtype': 'torch.bfloat16', 'input_channels_last': True, 'model_channels_last': True}],
           'optimizer_windows': [{'successful_updates': 25, 'gradient_updates_observed': 25, 'single_forward_per_update': True,
               'eligible_for_comparison': True, 'configuration_count': 1, 'ineligible_reasons': [],
               'backward_calls_per_update': [1] * 25,
               'wall_seconds': .5, 'examples_per_second': 3200., 'cuda_scalar_reads': {'item': 25, 'float': 0},
               'cuda_scalar_read_purposes': {'loss_scalar': 25, 'correctness': 0, 'unknown': 0},
               'training': {'batch': 64, 'autocast_dtype': 'torch.bfloat16', 'training_scope': 'full_model',
                            'input_channels_last': True, 'model_channels_last': True}}]}
    row.update(changes)
    return 'MLEVOLVE_RUNTIME_JSON ' + json.dumps(row)


def build(code=CODE, failure_text='', **changes):
    hardware = copy.deepcopy(next(n for n in GRAPH['nodes'] if n['id'] == IDS['5080']))
    edges = {e['to']: copy.deepcopy(e) for e in GRAPH['edges'] if e['from'] == IDS['5080']}
    return guidance_v7('improve', 'CNN image classification', code, DEVICES['5080'], hardware,
                       FEATURES, edges, RUNTIME, failure_text=failure_text, **changes)


def test_real_profile_changes_card_and_suppresses_already_configured_options():
    text, audit = build(failure_text=receipt())
    assert audit['matched_parent_receipt'] and audit['advice_ids'] == ['measured_sync']
    assert 'sm_120' in text and '25 source-attributed CUDA loss scalar' in text
    assert '[precision]' not in text and '[vision_layout]' not in text and '[memory]' not in text
    assert audit['decision_provenance'][0]['graph_patterns']
    assert audit['runtime_parent_window']['examples_per_second'] == 3200
    assert len(text) <= 1600


@pytest.mark.parametrize('changes', [{'code_sha256': 'wrong'}, {'execution_succeeded': False},
                                   {'gpu': {'name': 'NVIDIA A10', 'capability': [8, 6]}},
                                   {'first_step_finite': False}, {'origin': 'generated-code-print'}])
def test_mismatched_or_failed_receipt_cannot_drive_measured_advice(changes):
    text, audit = build(failure_text=receipt(**changes))
    assert not audit['matched_parent_receipt']
    assert not any(k.startswith('measured_') for k in audit['advice_ids'])


def test_no_new_intervention_means_no_added_tokens():
    text, audit = build()
    assert text is None and not audit['advice_ids']
    assert audit['source_configuration_is_runtime_proof'] is False


def test_successful_numeric_execution_with_bad_output_gets_no_hardware_debug_card():
    hardware = copy.deepcopy(next(n for n in GRAPH['nodes'] if n['id'] == IDS['5080']))
    edges = {e['to']: copy.deepcopy(e) for e in GRAPH['edges'] if e['from'] == IDS['5080']}
    text, audit = guidance_v7('debug', 'CNN image classification', CODE, DEVICES['5080'],
                              hardware, FEATURES, edges, RUNTIME,
                              failure_text='post_bn_recalibration val_accuracy=0.0103\n' + receipt())
    assert text is None and audit['matched_parent_receipt']
    assert not audit['advice_ids'] and not audit['decision_provenance']
    assert not audit['selected_feature_ids'] and not audit['verified_hardware_feature_ids']
    assert 'task/output debug' in audit['skip_reason']
    assert audit['policy_revision'] == 'v7.3'


def test_two_forward_sam_window_cannot_drive_single_forward_performance_advice():
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    row['optimizer_windows'][0]['single_forward_per_update'] = False
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert text is None and audit['runtime_parent_window'] is None
    assert not any(k.startswith('measured_') for k in audit['advice_ids'])


def test_batch_card_requires_actual_memory_and_window_evidence():
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    row['optimizer_windows'][0]['cuda_scalar_reads']['item'] = 0
    row['optimizer_windows'][0]['cuda_scalar_read_purposes']['loss_scalar'] = 0
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert audit['advice_ids'] == ['measured_batch']
    assert '16384 MiB' in text and '2048 MiB' in text and 'effective batch/update schedule' in text
    row['peak_allocated_bytes'] = 15 * 1024**3
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert text is None


def test_invalid_counters_or_multiple_optimizers_leave_throughput_unknown():
    for change in [{'window_validity': 'unknown'}, {'optimizer_windows': []}]:
        text, audit = build(failure_text=receipt(**change))
        assert text is None
        assert audit.get('runtime_parent_window') is None


def test_default_v7_draft_retains_verified_host_precision_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'hwkg/graph.json'))
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_SHA256', 'ef5850d7ab5af271413141ade5777878fbf8dcee0750e7eba406c924e4c6bec1')
    monkeypatch.setenv('MLEVOLVE_HWKG_HARDWARE_ID', IDS['5080'].removeprefix('hw:'))
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY', 'v7')
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR', str(tmp_path))
    with patch('hwkg.injection._visible_cuda_device', return_value=DEVICES['5080']), patch('hwkg.host_policy.runtime_facts', return_value=RUNTIME):
        text = hardware_guidance('draft', 'CNN image classification')
    row = json.loads((tmp_path / 'prompt-audit.jsonl').read_text())
    assert row['policy'] == 'hardware-v7' and row['advice_ids'] == ['precision']
    assert 'BF16' in text and 'feat:bf16' in row['selected_feature_ids']


def test_finite_checks_are_not_misrepresented_as_loss_logging():
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    row['optimizer_windows'][0]['cuda_scalar_read_purposes'] = {'loss_scalar': 0, 'correctness': 25}
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert 'measured_sync' not in audit['advice_ids']
    assert audit['scalar_read_attribution']['purposes']['correctness'] == 25
    del row['optimizer_windows'][0]['cuda_scalar_read_purposes']
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert 'measured_sync' not in audit['advice_ids']


def test_observed_fp32_overrides_unexecuted_bf16_syntax():
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    row['optimizer_windows'][0]['cuda_scalar_read_purposes']['loss_scalar'] = 0
    row['optimizer_windows'][0]['training'].update(autocast_dtype='disabled',
        input_channels_last=True, model_channels_last=True)
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert audit['advice_ids'] == ['precision']
    assert 'BF16' in text and audit['source_configuration_is_runtime_proof'] is False


def test_observed_inactive_layout_has_priority_over_capacity_reminder():
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    row['optimizer_windows'][0]['cuda_scalar_read_purposes']['loss_scalar'] = 0
    row['optimizer_windows'][0]['training'].update(autocast_dtype='torch.bfloat16',
        input_channels_last=False, model_channels_last=False)
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert audit['advice_ids'] == ['vision_layout'] and 'channels_last' in text


@pytest.mark.parametrize('case', ['legacy', 'mixed', 'head_only', 'unverified', 'unknown_contract', 'missing_rate', 'stale_gradient'])
def test_unverified_work_cannot_drive_measured_hardware_cards(case):
    row = json.loads(receipt().removeprefix('MLEVOLVE_RUNTIME_JSON '))
    window = row['optimizer_windows'][0]
    if case == 'legacy': row['schema'] = 'shared-runtime-v1'
    if case == 'mixed': window['configuration_count'] = 2
    if case == 'head_only': window['training']['training_scope'] = 'head_only'
    if case == 'unverified': window.pop('eligible_for_comparison')
    if case == 'unknown_contract': row['input_contract']['status'] = 'unknown'
    if case == 'missing_rate': window['examples_per_second'] = None
    if case == 'stale_gradient': window['backward_calls_per_update'] = [0] * 25
    text, audit = build(failure_text='MLEVOLVE_RUNTIME_JSON ' + json.dumps(row))
    assert audit['runtime_parent_window'] is None
    assert not any(k.startswith('measured_') for k in audit['advice_ids'])


def test_caught_input_violation_blocks_performance_advice():
    text, audit = build(failure_text=receipt(input_contract={'status': 'failed', 'violations': [{'input_shape': [64,3,96,96]}]}))
    assert text is None and not audit['matched_parent_receipt']
    assert 'runtime input contract violated' in audit['skip_reason']
