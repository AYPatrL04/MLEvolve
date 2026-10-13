"""Timing scope regressions. Replay only; never edit or execute candidates."""
import hashlib
from pathlib import Path
import pytest
from hwkg.window_scope import training_window_scope
from hwkg.native_evidence import native_feedback
from tests.test_hwkg_v8 import CODE, GPU, log, calibration, build

STANDALONE = CODE[:CODE.index('with torch.autocast')] + '''step=0
def train_step(xb,yb):
    global step
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=torch.nn.functional.cross_entropy(model(xb),yb)
    loss.backward()
    optimizer.step()
    step+=1
data_it=iter(loader)
window_start_step=step
for _ in range(20):
    xb,yb=next(data_it)
    train_step(xb,yb)
window_end_step=step
for epoch in range(EPOCHS):
    data_it=iter(loader)
    for xb,yb in data_it:
        train_step(xb,yb)
report={'window_start_step':window_start_step,'window_end_step':window_end_step}
'''
STATUS = 'standalone_gradient_window_before_epoch_iterator_reset_source_screen'
REASON = 'native timing window is a standalone gradient pass before epoch iterator reset'


def source(version):
    references = {
        'v88': ('hwkg-v88-native-5080-cifar100-local-s42-light4-20261013T151133Z',
                'fd37647268634519a8ae5c5d8404eea4', '5947ceec87dfb1e0e7928d053bb5d0c10ef706ae713e89837d75e3ad67a8c197'),
        'v89': ('hwkg-v89-native-5080-cifar100-local-s42-light4-postpoweron-20261013T185732Z',
                '5ae7b821dbc94d529520c8a8eadf7e82', 'b12b90a41c1b393a1ea93ad1b01b0ef6e174f049e9d2ad0689b890e2178eece1')}
    root, node, sha = references[version]
    folder = Path(__file__).resolve().parents[2]/'MLEvolve/records'/root/'treatment/candidates'/node
    if not folder.exists(): pytest.skip('saved public source unavailable')
    code = (folder/'candidate.py').read_text()
    assert hashlib.sha256(code.encode()).hexdigest() == sha
    return folder, code


def test_actual_separate_window_is_annotated_without_grading_the_candidate():
    folder, code = source('v88')
    result = native_feedback(code, (folder/'execution.log').read_text(), GPU,
                             parent_id=folder.name, execution_succeeded=True)
    assert result['window']['normal_training_source_screen']['status'] == STATUS
    assert REASON in result['reasons']
    scope = result['window']['normal_training_source_screen']
    assert not scope['epoch_budget_verified'] and not scope['runtime_verified']
    assert not result['verified_throughput']


def test_actual_embedded_window_remains_self_report_only():
    folder, code = source('v89')
    scope = training_window_scope(code)
    assert scope['status'] == 'embedded_epoch_window_source_screen'
    assert not scope['runtime_verified'] and not scope['epoch_budget_verified']
    result = native_feedback(code, (folder/'execution.log').read_text(), GPU,
                             parent_id=folder.name, execution_succeeded=True)
    assert REASON not in result['reasons']
    assert 'native FP16 GradScaler successful-work counters not established by source' in result['reasons']


def test_counter_consistency_is_insufficient_without_training_scope():
    result = native_feedback(STANDALONE, log(), GPU, parent_id='p', execution_succeeded=True)
    assert result['window'] and result['trust'] == 'source_bound_self_report'
    assert result['source_spec']['training_scope'] == 'full_model_source_screen'
    assert result['reasons'] == [REASON] and not result['verified_throughput']


@pytest.mark.parametrize('old,new', [('train_step','update_batch'), ('data_it','batches'),
    ('window_start_step=step','window_start_step=counter'), ('range(20)','range(27)'), ('EPOCHS=6','EPOCHS=2')])
def test_structure_not_candidate_names_or_fixed_epoch_budget(old, new):
    scope = training_window_scope(STANDALONE.replace(old, new))
    assert scope['status'] == STATUS
    assert not scope['epoch_budget_verified']


@pytest.mark.parametrize('changed', [
    STANDALONE.replace('    data_it=iter(loader)\n', ''),
    STANDALONE.replace('    data_it=iter(loader)\n', '    data_it=iter(data_it)\n'),
    STANDALONE.replace("'window_start_step':window_start_step", "'window_start_step':0"),
    STANDALONE.replace('window_start_step=step', 'window_start_step=step\nwindow_start_step=other'),
    STANDALONE.replace('window_end_step=step', 'window_end_step=step\nwindow_end_step=other'),
    STANDALONE+"\nreport2={'window_start_step':window_start_step,'window_end_step':window_end_step}",
    STANDALONE.replace('data_it=iter(loader)\nwindow', 'train_step=other\ndata_it=iter(loader)\nwindow'),
    STANDALONE.replace('data_it=iter(loader)\nwindow', 'from other import train_step\ndata_it=iter(loader)\nwindow'),
])
def test_ambiguous_or_continuation_forms_are_unknown(changed):
    assert training_window_scope(changed)['status'] == 'unknown'


def test_logging_only_interval_does_not_become_gradient_work():
    code = STANDALONE.replace('    train_step(xb,yb)\nwindow_end_step', '    print(xb.shape)\nwindow_end_step')
    assert training_window_scope(code)['status'] == 'unknown'


def test_wrapped_and_broken_sources_are_explicit_unknown():
    assert training_window_scope('```python\n'+STANDALONE)['status'] == 'unknown'
    assert training_window_scope('')['status'] == 'unknown'
    result = native_feedback(CODE, log(window_start_step=0, window_end_step=20), GPU,
                             parent_id='p', execution_succeeded=True)
    assert result['window']['normal_training_source_screen']['status'] == 'unknown'
    assert REASON not in result['reasons']


def test_standalone_scope_blocks_projection_not_upstream_scoring(tmp_path, monkeypatch):
    _, record = calibration(tmp_path)
    from hwkg.calibration import choose_trial
    assert choose_trial(record, 'bf16_contiguous', 3750, 5)
    monkeypatch.setattr('hwkg.feedback_policy.select_record', lambda *a: (record, 'exact matched fixture'))
    _, audit = build(code=STANDALONE)
    assert audit['calibration_trial'] is None
    assert audit['native_parent_evidence']['reasons'] == [REASON]
    assert not audit['end_to_end_gain_verified']


def test_cyclic_helper_calls_terminate_and_unreached_helpers_do_not_count():
    code = 'def a():b()\ndef b():a()\n' + STANDALONE.replace('    train_step(xb,yb)\nwindow_end_step', '    a()\nwindow_end_step')
    assert training_window_scope(code)['status'] == 'unknown'
