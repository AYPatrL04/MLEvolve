"""Actual native counter failures; source/log replay only, no training edits."""
import hashlib
import json
from pathlib import Path
import pytest
from hwkg.native_evidence import source_spec, native_feedback, scaler_update_accounting
from tests.test_hwkg_v8 import CODE, GPU, log, read


def actual_source(version, candidate, expected):
    roots = {
        'v88': 'hwkg-v88-native-5080-cifar100-local-s42-light4-20261013T151133Z',
        'v89': 'hwkg-v89-native-5080-cifar100-local-s42-light4-postpoweron-20261013T185732Z',
    }
    folder = Path(__file__).resolve().parents[2]/'MLEvolve/records'/roots[version]/'treatment/candidates'/candidate
    if not folder.exists(): pytest.skip('local historical source fixture absent')
    code = (folder/'candidate.py').read_text()
    assert hashlib.sha256(code.encode()).hexdigest() == expected
    return folder, code


def test_actual_tuple_batch_checks_native_example_count():
    folder, code = actual_source('v88', 'fd37647268634519a8ae5c5d8404eea4', '5947ceec87dfb1e0e7928d053bb5d0c10ef706ae713e89837d75e3ad67a8c197')
    assert source_spec(code)['batch'] == 128
    assert read(log(completed_examples=1280), code)['window'] is None
    assert 'native examples disagree with source batch' in read(log(), code)['reasons']
    # Coverage does not relax custom model/loss/groups into a direct calibration.
    assert source_spec(code)['model'] is None
    assert source_spec(code)['optimizer_options'] is None


def test_actual_fp16_scaler_attempts_are_not_successful_work_proof():
    folder, code = actual_source('v89', '5ae7b821dbc94d529520c8a8eadf7e82', 'b12b90a41c1b393a1ea93ad1b01b0ef6e174f049e9d2ad0689b890e2178eece1')
    result = native_feedback(code, (folder/'execution.log').read_text(), GPU, parent_id=folder.name, execution_succeeded=True)
    assert result['window'] and result['trust'] == 'source_bound_self_report'
    assert result['reasons'] == ['native FP16 GradScaler successful-work counters not established by source']
    assert not result['verified_throughput']


@pytest.mark.parametrize('binding', ['BATCH,NW=64,4', '(BATCH,NW)=(64,4)', '[BATCH,NW]=[64,4]', 'BATCH: int=64', 'BASE=64\nBATCH=BASE', 'BASE=64\nALIAS=BASE\nBATCH,NW=ALIAS,4'])
def test_declarative_batch_forms_bind_counter_math(binding):
    code = CODE.replace('BATCH=64', binding)
    assert source_spec(code)['batch'] == 64
    assert read(log(completed_examples=640,examples_per_second=1280), code)['window'] is None


@pytest.mark.parametrize('binding', [
    'BATCH=64\nBATCH=128', 'BATCH=64\nBATCH+=64',
    'BATCH=64\nif flag:BATCH=128', 'BATCH=64\ndef f(BATCH):return BATCH',
    'BATCH=64\ndef f():\n global BATCH\n BATCH=128',
    'BATCH=64\ntry:run()\nexcept Exception as BATCH:pass',
    'BATCH=64\nmatch value:\n case {"batch": BATCH}:pass',
    'BATCH=int("64")', 'BATCH=RATE\nRATE=BATCH',
    'BATCH,NW=get_settings()', 'BATCH,NW=64,4\nBS=unknown',
    'if flag:BATCH=64', 'BATCH,NW=64,4\ndel BATCH',
])
def test_dynamic_shadowed_rebound_batch_is_unknown(binding):
    assert source_spec(CODE.replace('BATCH=64',binding))['batch'] is None


def test_unresolved_shape_alias_does_not_borrow_dimensions():
    code = CODE.replace('SIZE=128','SIDE=128\nSIZE=SIDE')
    assert source_spec(code)['input_shape'] == [3,128,128]
    assert source_spec(code+'\nSIDE=256')['input_shape'] is None


def scaler_code(enabled='True', accounting=''):
    return CODE.replace('torch.bfloat16','torch.float16') + f"\nscaler=torch.amp.GradScaler('cuda',enabled={enabled})\nscaler.scale(loss).backward()\nscaler.step(optimizer)\nscaler.update()\n" + accounting


def test_literal_skip_zero_without_scale_count_blocks_matched_window():
    code = scaler_code(accounting="thr={'optimizer_steps':20,'skipped_updates':0}")
    result = read(log(actual_autocast_dtype='float16'),code)
    assert result['reasons'] and not result['verified_throughput']


def test_explicit_disabled_scaler_is_distinct_from_dynamic_enabled():
    assert scaler_update_accounting(scaler_code('False')) == 'no_active_scaler_source_screen'
    assert scaler_update_accounting(scaler_code('use_amp')) == 'successful_updates_unestablished'


def test_linked_scale_delta_is_only_a_syntax_screen():
    code = scaler_code(accounting="""
before=scaler.get_scale()
scaler.step(optimizer)
scaler.update()
after=scaler.get_scale()
if after < before:
    skipped+=1
else:
    updates+=1
thr={'optimizer_steps':updates,'skipped_updates':skipped}
""")
    assert scaler_update_accounting(code) == 'scale_delta_counter_syntax_only'
    result = read(log(actual_autocast_dtype='float16'),code)
    assert not result['verified_throughput']
    assert scaler_update_accounting(code.replace('skipped+=1','other+=1')) == 'successful_updates_unestablished'
    assert scaler_update_accounting(code.replace('after < before','after < after')) == 'successful_updates_unestablished'
    assert scaler_update_accounting(code.replace("'skipped_updates':skipped","'skipped_updates':0")) == 'successful_updates_unestablished'


def test_disabled_or_bf16_report_remains_uncertified():
    result=read(log(actual_autocast_dtype='float16'),scaler_code('False'))
    assert not result['reasons'] and not result['verified_throughput']
    assert not read()['verified_throughput']


def test_mutable_optimizer_setting_is_not_a_declarative_constant():
    code=CODE.replace('weight_decay=1e-4', 'weight_decay=1e-4,betas=BETAS')
    assert source_spec('BETAS=(.9,.999)\n'+code)['optimizer_options']['betas']==[.9,.999]
    assert source_spec('BETAS=[.9,.999]\nBETAS[0]=.5\n'+code)['optimizer_options'] is None


def test_aliased_scaler_does_not_become_no_scaler_evidence():
    code=scaler_code().replace('torch.amp.GradScaler', 'GS')
    assert scaler_update_accounting('from torch.amp import GradScaler as GS\n'+code)=='unknown_scaler_binding'


def test_native_fp16_counter_failure_suppresses_calibration_projection(tmp_path,monkeypatch):
    from tests.test_hwkg_v8 import calibration,build
    path,row=calibration(tmp_path)
    row['profiles']['fp16_contiguous']={**row['profiles']['fp32_contiguous'],'step_seconds':.02}
    from hwkg.calibration import choose_trial
    assert choose_trial(row,'fp16_contiguous',3750,5) is not None
    # Select a known matched fixture explicitly; isolate the evidence gate.
    monkeypatch.setattr('hwkg.feedback_policy.select_record',lambda *a:(row,'exact matched fixture'))
    code=scaler_code(accounting="thr={'optimizer_steps':20,'skipped_updates':0}")
    _,audit=build(code=code,failure_text=log(actual_autocast_dtype='float16'))
    assert audit['native_parent_evidence']['reasons']
    assert audit['calibration_trial'] is None
    assert audit['calibration_basis']=='native work scope cannot support matched full-model calibration'


def test_dynamic_scaler_factory_cannot_establish_absence():
    code=scaler_code().replace("torch.amp.GradScaler('cuda',enabled=True)", 'factory()')
    assert scaler_update_accounting(code)=='unknown_scaler_binding'
