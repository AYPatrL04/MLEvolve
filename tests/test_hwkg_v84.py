"""Regressions for actual native failures; all fixes remain advice/observations."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg.source_analysis import precision_scope
from hwkg.native_evidence import source_spec, recipe_changes
from hwkg.calibration import choose_trial, select_record, workload_key, SCHEMA
from hwkg.trial_memory import remember, observe, _pending, _blocked
from tests.test_hwkg_v8 import CODE, GPU, RUNTIME, TASK, build, log, calibration


def test_validation_and_metric_detach_do_not_poison_training_scope():
    code=CODE+'''\n@torch.no_grad()
def predict(x):
    return model(x).detach()
with torch.no_grad():
    metric_value=logits.detach().sum()
'''
    assert source_spec(code)['training_scope']=='full_model_source_screen'
    assert source_spec(code.replace('loss.backward()', 'loss=loss.detach()\nloss.backward()'))['training_scope']=='unknown'
    assert source_spec(CODE.replace("with torch.autocast('cuda',dtype=torch.bfloat16):",'with torch.no_grad():'))['training_scope']=='unknown'


def test_cross_entropy_class_matches_only_default_recipe():
    code=CODE.replace('torch.nn.functional.cross_entropy(logits,yb)', 'criterion(logits,yb)')
    assert source_spec(code+'\ncriterion=torch.nn.CrossEntropyLoss()')['loss']=='cross_entropy_default'
    assert source_spec(code+'\ncriterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)')['loss']['label_smoothing']==.1


def test_literal_backend_override_matches_actual_calibration(tmp_path):
    path,row=calibration(tmp_path)
    code=CODE+'\ntorch.backends.cudnn.benchmark=True'
    spec=source_spec(code)
    assert spec['runtime_overrides']=={'cudnn_benchmark':True}
    row['workload']=workload_key(GPU,RUNTIME,spec)
    path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}))
    assert select_record(path,GPU,RUNTIME,spec)[0]
    assert select_record(path,GPU,RUNTIME,source_spec(code.replace('=True','=flag')))[0] is None


def test_fp16_and_layout_single_change_with_cost_reserve(tmp_path):
    _,row=calibration(tmp_path)
    row['profiles']['fp16_contiguous']={**row['profiles']['bf16_contiguous'],'step_seconds':.009}
    row['profiles']['fp16_channels_last']={**row['profiles']['bf16_contiguous'],'step_seconds':.003}
    assert choose_trial(row,'fp16_contiguous',1000,1)['profile']=='fp16_channels_last'
    assert choose_trial(row,'fp16_channels_last',1000,1) is None
    assert choose_trial(row,'fp16_contiguous',1000,100) is None
    assert choose_trial(row,'fp16_contiguous',1000,1,['fp16_channels_last']) is None


def test_unmatched_precision_option_is_explicit_hypothesis_not_calibration_gain():
    text,audit=build(code=CODE.replace('bfloat16','float16')+'\nmodel.to(memory_format=torch.channels_last)',
        failure_text=log(actual_autocast_dtype='float16',model_memory_format='channels_last',input_memory_format='channels_last'))
    assert text and audit['current_profile']=='fp16_channels_last'
    assert audit['proposal_kind']=='capability_guided_unmeasured'
    assert audit['calibration_trial'] is None and not audit['measurements_verified']
    assert 'TRAIN-only torch.bfloat16' in text and 'Preserve shared eval precision' in text
    assert audit['added_provider_calls']==0


def test_inclusive_and_exclusive_windows_are_recorded():
    from tests.test_hwkg_v8 import read
    assert read(log(window_end_step=29))['window']['window_index_convention']=='inclusive'
    assert read(log())['window']['window_index_convention']=='exclusive'
    assert read(log(window_end_step=28))['window'] is None


def test_actual_shared_helper_drift_and_scaler_detected():
    root=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkg-v83-native-5080-cifar100-local-s42-15plus15-reverse-20261013T041812Z/treatment/candidates'
    if not root.exists(): pytest.skip('historical regression fixture not present')
    before=(root/'d5f637a38ecd4b26a6b8222450a09eb5/candidate.py').read_text()
    after=(root/'0084dbda53d24e218d757d58cdaeb956/candidate.py').read_text()
    audit=recipe_changes(before,after)
    assert audit['training_precision_changed'] and audit['evaluation_precision_changed']
    assert audit['bf16_with_active_scaler'] and not audit['semantic_preservation_certified']
    assert source_spec(before)['optimizer_options'] is None  # dynamic parameter groups stay unknown


def test_train_only_helper_change_preserves_eval_signature():
    before=CODE.replace("with torch.autocast('cuda',dtype=torch.bfloat16):",'with train_amp():')+'''\ndef train_amp():
    return torch.autocast('cuda',dtype=torch.float16)
@torch.no_grad()
def predict(x):
    with torch.autocast('cuda',dtype=torch.float16):
        return model(x)
'''
    after=before.replace("return torch.autocast('cuda',dtype=torch.float16)", "return torch.autocast('cuda',dtype=torch.bfloat16)")
    audit=recipe_changes(before,after)
    assert audit['training_precision_changed'] and not audit['evaluation_precision_changed']


def test_negative_trial_only_suppresses_future_advice_and_is_run_local(monkeypatch):
    _pending.clear();_blocked.clear()
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR','unique-run-1')
    parent=NS(id='p',code=CODE,exec_time=10.,metric=NS(value=.85,maximize=True))
    child=NS(id='c',code=CODE.replace('bfloat16','float16'),exec_time=None,is_buggy=None,metric=NS(value=.84,maximize=True))
    remember(TASK,parent,child,{'precision'},'fp16_contiguous')
    assert observe(TASK,parent.code)==([],[])
    child.exec_time=12.;child.is_buggy=False
    frozen=copy.deepcopy(vars(child))
    blocked,events=observe(TASK,parent.code)
    assert blocked==['fp16_contiguous'] and events[0]['suppression_reasons']
    assert vars(child)==frozen and not events[0]['node_modified']
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR','unique-run-2')
    assert observe(TASK,parent.code)==([],[])


@pytest.mark.parametrize('reserve',['nan','-1','bad'])
def test_invalid_cost_config_fails_closed(reserve,monkeypatch):
    monkeypatch.setenv('MLEVOLVE_HWKG_COST_RESERVE_SECONDS',reserve)
    assert build()[0] is None


def test_profitable_fp16_trial_is_actionable_and_format_bounded(tmp_path,monkeypatch):
    path,row=calibration(tmp_path)
    row['profiles']['fp16_contiguous']={**row['profiles']['bf16_contiguous'],'step_seconds':.001}
    row['profiles']['bf16_channels_last']['step_seconds']=.011
    path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}))
    monkeypatch.setenv('MLEVOLVE_HWKG_CALIBRATION_PATH',str(path))
    text,audit=build()
    assert text and len(text)<=1100 and audit['calibration_trial']['profile']=='fp16_contiguous'
    assert 'torch.float16' in text and 'Use GradScaler' in text and 'shared eval autocast helper' in text
    assert audit['advice_ids']==['precision'] and not audit['measurements_verified']
    assert build(blocked_profiles=['fp16_contiguous'])[0] is None
