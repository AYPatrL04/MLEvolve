"""Actual native parent calibration gaps and duplicate host fact regressions."""
from pathlib import Path
import copy
import json
import pytest
from hwkg.native_evidence import source_spec
from hwkg.calibration import workload_key, select_record, choose_trial
from hwkg.routing import hardware_context, with_hardware_prompt
from types import SimpleNamespace as NS
from tests.test_hwkg_v8 import CODE, TASK, GPU, RUNTIME, build, enabled
from tests.test_hwkg_v6 import CFG


def parent_fixture():
    root=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkg-v86-native-5080-cifar100-local-s42-medium5plus5-20261013T125406Z'
    if not root.exists(): pytest.skip('observed public regression source absent')
    return (root/'treatment/candidates/eb990f5a4b19400d99d32d8b63598f82/candidate.py').read_text(), (root/'data/description.md').read_text()


def test_actual_smoothed_loss_literal_shape_and_optimizer_key_are_complete():
    code,_=parent_fixture();spec=source_spec(code)
    assert spec['input_shape']==[3,128,128]
    assert spec['loss']=={'kind':'cross_entropy','label_smoothing':.1,'reduction':'mean'}
    assert spec['optimizer_options']['lr']==.0003 and spec['optimizer_options']['weight_decay']==.05
    assert spec['forwards_per_update']==1 and spec['training_scope']=='full_model_source_screen'


@pytest.mark.parametrize('bad', ['size=shape', 'scale_factor=4', 'size=(128,64)'])
def test_unknown_or_non_square_resize_does_not_borrow_literal_allocation(bad):
    code=CODE.replace('SIZE=128\n','')+'\nx=F.interpolate(x, size=128)\nout=torch.empty((N,3,128,128))'
    code=code.replace('import torch','import torch\nimport torch.nn.functional as F')
    assert source_spec(code)['input_shape']==[3,128,128]
    assert source_spec(code.replace('size=128',bad))['input_shape'] is None
    assert source_spec(code+'\nz=F.interpolate(z,size=224)')['input_shape'] is None
    assert source_spec(code.replace('(N,3,128,128)','(N,3,64,64)'))['input_shape'] is None


@pytest.mark.parametrize('value', ['smoothing', 'True', '-.1', '1.1', 'float("nan")'])
def test_unresolved_or_invalid_smoothing_stays_unknown(value):
    assert source_spec(CODE.replace('logits,yb','logits,yb,label_smoothing='+value))['loss'] is None


def test_weighted_or_other_reduction_does_not_match_default_loss():
    assert source_spec(CODE.replace('logits,yb','logits,yb,weight=w'))['loss'] is None
    assert source_spec(CODE.replace('logits,yb','logits,yb,reduction="sum"'))['loss'] is None
    assert source_spec(CODE.replace('logits,yb','logits,yb,label_smoothing=0'))['loss']=='cross_entropy_default'


def test_literal_tf32_assignment_couples_precision_only_in_verified_runtime(tmp_path):
    code,_=parent_fixture();spec=source_spec(code)
    runtime={**RUNTIME,'torch':'2.14.1+cu130','cuda':'13.0','calibration_flags':{**RUNTIME['calibration_flags'],'float32_matmul_precision':'highest'}}
    key=workload_key(GPU,runtime,spec)
    assert key['runtime_flags']['matmul_allow_tf32'] and key['runtime_flags']['float32_matmul_precision']=='high'
    row={'workload':key,'quality_verified':False,'profiles':{'bf16_channels_last':{'step_seconds':.05,'setup_seconds':1.,'repeats':3,'normal_updates':25,'finite_updates':True,'finite_check_scope':'all_losses_initial_grad_final_parameters'}}}
    path=tmp_path/'key.json';path.write_text(json.dumps({'schema':'hwkg-calibration-v1','records':[row]}))
    assert select_record(path,GPU,runtime,spec)[0]
    assert select_record(path,GPU,{**runtime,'torch':'unverified'},spec)[0] is None
    assert workload_key(GPU,runtime,source_spec(code.replace('matmul.allow_tf32 = True','matmul.allow_tf32 = False')))['runtime_flags']['float32_matmul_precision']=='highest'


def test_native_task_hardware_facts_are_not_repeated_on_draft(enabled):
    _,task=parent_fixture();sent=[]
    @with_hardware_prompt
    def provider(prompt,cfg,**kwargs):sent.append((prompt,cfg,kwargs));return 'original provider response'
    @hardware_context('draft')
    def draft(agent):return provider(task+'\n# Response format\nOriginal format',CFG,max_tokens=16384)
    assert draft(NS(task_desc=task))=='original provider response'
    assert sent==[(task+'\n# Response format\nOriginal format',CFG,{'max_tokens':16384})]
    rows=[json.loads(l) for l in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    assert rows[-1]['chars']==0 and 'already supplies' in rows[-1]['skip_reason']
    assert build(stage='draft',code='',parent_execution_succeeded=None,failure_text='')[0]  # absent facts remain available


def test_sam_and_parameter_groups_stay_outside_direct_calibration():
    code,_=parent_fixture()
    assert source_spec(code.replace('optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)', 'optimizer = SAM(model.parameters(),torch.optim.AdamW,rho=.05,lr=3e-4,weight_decay=.05)'))['optimizer_options'] is None
    assert source_spec(code.replace('model.parameters(), lr=3e-4','[{"params":model.parameters(),"lr":3e-4}], lr=3e-4'))['optimizer_options'] is None


def test_tiny_median_profit_cannot_ignore_recorded_repeat_spread():
    record={'profiles':{
        'bf16_channels_last':{'step_seconds':.061,'step_seconds_min':.060,'step_seconds_max':.063,'setup_seconds':.1},
        'fp16_channels_last':{'step_seconds':.0555,'step_seconds_min':.055,'step_seconds_max':.057,'setup_seconds':.1}}}
    assert choose_trial(record,'bf16_channels_last',936,5) is None
    trial=choose_trial(record,'bf16_channels_last',2000,5)
    assert trial['repeat_spread_net_seconds']>0 and 'not statistical CI' in trial['uncertainty_basis']
    assert not trial['quality_verified'] and not trial['end_to_end_verified']


def test_actual_observed_record_matches_parent_but_blocks_marginal_precision_swap():
    from hwkg.calibration import calibration_path, calibration_flags
    from hwkg.host_policy import runtime_facts
    code,_=parent_fixture();spec=source_spec(code)
    runtime={**runtime_facts(), 'calibration_flags':calibration_flags()}
    record,basis=select_record(calibration_path(GPU),GPU,runtime,spec)
    assert record and record['workload']['loss']['label_smoothing']==.1
    assert record['profiles']['bf16_channels_last']['repeats']==3
    assert choose_trial(record,'bf16_channels_last',936,5) is None
    assert choose_trial(record,'bf16_contiguous',936,5)['change']=='layout'
