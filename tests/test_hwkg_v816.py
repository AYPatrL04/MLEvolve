"""Capability hypotheses are delivered without manufacturing calibration gains."""
import copy
import json
from types import SimpleNamespace as NS
import pytest
from hwkg.exploration_policy import guidance,optimizer_option
from hwkg.native_evidence import source_spec
from hwkg.feedback_policy import guidance_v8
from hwkg.routing import hardware_context,with_hardware_prompt
from tests.test_hwkg_v814 import CODE, EDGES, RUNTIME, GPU
from tests.test_hwkg_v8 import build, TASK, enabled
from tests.test_hwkg_v6 import CFG

R={**RUNTIME,'adamw_fused_api':True,'grad_scaler_api':True}
GROUPED=CODE.replace('model.parameters(),lr=0.0003',"[{'params':model.parameters(),'lr':.002}]")

def advice(code=GROUPED,**kw):
    return guidance(kw.pop('stage','improve'),code,kw.pop('device',GPU),kw.pop('edges',EDGES),
        kw.pop('runtime',R),phase=kw.pop('phase','coder'),
        parent_execution_succeeded=kw.pop('success',True),**kw)


def test_initial_patterns_survive_already_present_host_identity():
    text,audit=advice('',stage='draft',success=None,identity_already_present=True)
    assert len(text)<=340 and 'BF16' in text and 'fused=True' in text
    assert audit['draft_identity_facts_deduplicated'] and audit['draft_recipe_forced'] is False
    assert audit['advice_ids']==['precision_choice','optimizer_choice']
    assert audit['calibration_trial'] is None and not audit['measurements_verified']


@pytest.mark.parametrize('phase',['planner','coder','plan_and_code'])
def test_grouped_or_uncalibrated_parent_gets_one_implementation_hypothesis(phase):
    result=advice(phase=phase);assert result
    text,audit=result
    assert len(text)<=750 and 'groups/LR/WD' in text and 'fused=True' in text
    assert 'no gain projection' in text and 'no extra benchmark' in text
    assert audit['proposal_kind']=='capability_guided_unmeasured'
    assert audit['calibration_trial'] is None and not audit['end_to_end_gain_verified']
    assert not audit['declared_or_measured_cost_gate_applied']


def test_failed_fused_profile_allows_supported_separate_FP16_hypothesis():
    text,audit=advice(blocked_profiles=['optimizer_fused_adamw'])
    assert 'TRAIN-only torch.float16' in text and 'GradScaler' in text
    assert 'eval dtype separate' in text and 'AdamW unfused' in text
    assert audit['advice_ids']==['precision'] and audit['exploration_trial']['profile']=='fp16_channels_last'


@pytest.mark.parametrize('kw',[{'stage':'debug'},{'phase':'merge'},{'success':False},{'success':None},
    {'device':NS(name='NVIDIA A10',major=8,minor=6,total_memory=24*1024**3)},
    {'max_chars':30},{'blocked_profiles':['optimizer_fused_adamw','fp16_channels_last']}])
def test_no_forced_irrelevant_failed_or_overbudget_proposal(kw):
    assert advice(**kw) is None


@pytest.mark.parametrize('suffix',[
    ',fused=True',',fused=ENABLED',',foreach=True',',differentiable=True',',capturable=True',
])
def test_already_fused_unknown_or_incompatible_options_not_recommended(suffix):
    code=CODE.replace('weight_decay=0.05)',f'weight_decay=0.05{suffix})')
    assert advice(code) is None


def test_FP16_fused_domain_with_observed_scale_skip_is_not_offered():
    text,audit=advice(CODE.replace('bfloat16','float16'))
    assert audit['advice_ids']==['precision'] and 'fused=True' not in text
    assert 'torch.bfloat16' in text and 'disable TRAIN GradScaler' in text
    assert advice('import numpy as np\nx=np.zeros(10)') is None


def test_required_host_edge_and_API_supported_are_distinct_from_speed():
    e=copy.deepcopy(EDGES);e['feat:optimizer_fused_adamw']['properties']['verified']=False
    assert advice(edges=e,runtime={**R,'grad_scaler_api':False}) is None
    assert advice(runtime={**R,'adamw_fused_api':False,'grad_scaler_api':False}) is None
    assert advice(runtime={**R,'bf16_supported':False,'adamw_fused_api':False,'grad_scaler_api':False}) is None


def test_no_optimizer_class_replacement_or_duplicate_claim():
    assert optimizer_option(CODE)['fused'] is None
    assert optimizer_option(GROUPED)['fused'] is None
    assert optimizer_option(CODE+'\nother=torch.optim.AdamW(model.parameters())') is None
    assert optimizer_option(CODE.replace('torch.optim.AdamW','SAM')) is None
    assert optimizer_option('this is not Python') is None


def test_policy_Fallback_does_not_borrow_exact_measured_record():
    text,audit=build(code=GROUPED,runtime=R,failure_text='',phase='coder')
    assert text and audit['proposal_kind']=='capability_guided_unmeasured'
    assert audit['calibration_trial'] is None and audit['advice_ids']==['optimizer_fused_adamw']
    assert not audit['measurements_verified'] and not audit['end_to_end_gain_verified']


def test_provider_receives_patterns_with_original_budget_and_returns_unchanged(enabled,monkeypatch):
    monkeypatch.setattr('hwkg.host_policy.runtime_facts',lambda:R)
    full={'system':'system','user':'Task\n# Response format\noriginal fences',
          'assistant':'NVIDIA GeForce RTX 5080 capability 12.0 torch2.14.1+cu130'}
    frozen=copy.deepcopy(full);calls=[]
    @with_hardware_prompt
    def provider(prompt,cfg,**kwargs):calls.append((prompt,cfg,kwargs));return 'native answer'
    @hardware_context('draft')
    def run(agent):return provider(full,CFG,max_tokens=16384,temperature=1.)
    assert run(NS(task_desc=TASK))=='native answer' and full==frozen
    delivered,cfg,kw=calls[0]
    assert '[precision_choice]' in delivered['user'] and '[optimizer_choice]' in delivered['user']
    assert delivered['system']==frozen['system'] and delivered['assistant']==frozen['assistant']
    assert cfg is CFG and kw=={'max_tokens':16384,'temperature':1.}
    rows=[json.loads(l) for l in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    assert rows[0]['chars']>0 and rows[0]['added_provider_calls']==0


@pytest.mark.parametrize('backward',['loss.backward()','scaler.scale(loss).backward()'])
def test_known_active_loss_constructor_can_match_without_certifying_semantics(backward):
    code=CODE.replace('loss=torch.nn.functional.cross_entropy(logits,yb,label_smoothing=0.1)',
        'criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)\nloss=criterion(logits,yb)').replace('loss.backward()',backward)
    assert source_spec(code)['loss']['label_smoothing']==.1


@pytest.mark.parametrize('name',['other_loss','mix_loss','arbitrary_helper'])
def test_unused_CE_constructor_does_not_certify_custom_actual_training_loss(name):
    code=CODE.replace('loss=torch.nn.functional.cross_entropy(logits,yb,label_smoothing=0.1)',
        f'unused_criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)\nloss={name}(logits,yb)')
    assert source_spec(code)['loss'] is None
    # Capability-only optimizer hypothesis does not claim a CE timing projection.
    result=advice(code);assert result and result[1]['calibration_trial'] is None
