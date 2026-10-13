"""Scoped module dtype aliases and quality-confounded feedback regressions."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg.source_analysis import precision_scope
from hwkg.native_evidence import recipe_changes
from hwkg import trial_memory


def actual_source():
    path=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkg-v88-native-5080-cifar100-local-s42-light4-20261013T151133Z/treatment/candidates/fd37647268634519a8ae5c5d8404eea4/candidate.py'
    if not path.exists():pytest.skip('local historical source absent')
    code=path.read_text()
    assert hashlib.sha256(code.encode()).hexdigest()=='5947ceec87dfb1e0e7928d053bb5d0c10ef706ae713e89837d75e3ad67a8c197'
    return code


def sample(binding='AMP_DTYPE=torch.bfloat16'):
    return 'import torch\n'+binding+"""
@torch.no_grad()
def predict(x):
    with torch.autocast('cuda',dtype=AMP_DTYPE):return model(x)
with torch.autocast('cuda',dtype=AMP_DTYPE):loss=model(x).sum()
loss.backward()
optimizer.step()
"""


def test_actual_shared_dtype_change_is_detected_in_both_scopes():
    code=actual_source();scope=precision_scope(code)
    assert scope['training_dtypes']==['bfloat16']
    changed=code.replace('AMP_DTYPE = torch.bfloat16','AMP_DTYPE = torch.float16')
    audit=recipe_changes(code,changed)
    assert audit['training_precision_changed'] and audit['evaluation_precision_changed']
    assert not audit['semantic_preservation_certified']


@pytest.mark.parametrize('binding',['AMP_DTYPE: object=torch.bfloat16','BASE=torch.bfloat16\nAMP_DTYPE=BASE','BASE=torch.bfloat16\nAMP_DTYPE,NW=BASE,4'])
def test_unique_declarative_dtype_aliases(binding):
    assert precision_scope(sample(binding))['training_dtypes']==['bfloat16']


@pytest.mark.parametrize('binding',[
    'AMP_DTYPE=torch.bfloat16\nAMP_DTYPE=torch.float16',
    'if flag:AMP_DTYPE=torch.bfloat16',
    'AMP_DTYPE=choose_dtype()',
    'AMP_DTYPE=OTHER\nOTHER=AMP_DTYPE',
    'AMP_DTYPE=torch.bfloat16\ndef f(AMP_DTYPE):return AMP_DTYPE',
    'AMP_DTYPE=torch.bfloat16\nclass torch:pass',
    'AMP_DTYPE=other.bfloat16',
])
def test_dynamic_shadowed_or_non_torch_dtype_remains_unknown(binding):
    assert precision_scope(sample(binding))['training_dtypes']==[]


def test_torch_namespace_import_alias_is_source_syntax_only():
    code=sample().replace('import torch','import torch as T').replace('torch.','T.')
    assert precision_scope(code)['training_dtypes']==['bfloat16']
    assert precision_scope(code)['syntax_only']


def test_train_only_alias_preserves_eval_signature():
    code=sample('TRAIN_DTYPE=torch.bfloat16\nEVAL_DTYPE=torch.float32')
    # Intentionally distinct aliases in train vs eval.
    code=code.replace('dtype=AMP_DTYPE):return','dtype=EVAL_DTYPE):return').replace('dtype=AMP_DTYPE):loss','dtype=TRAIN_DTYPE):loss')
    changed=code.replace('TRAIN_DTYPE=torch.bfloat16','TRAIN_DTYPE=torch.float16')
    audit=recipe_changes(code,changed)
    assert audit['training_precision_changed'] and not audit['evaluation_precision_changed']


def test_enabled_alias_changes_both_precision_scope_signatures():
    code=sample('AMP_DTYPE=torch.bfloat16\nAMP_ENABLED=True').replace('dtype=AMP_DTYPE','dtype=AMP_DTYPE,enabled=AMP_ENABLED')
    audit=recipe_changes(code,code.replace('AMP_ENABLED=True','AMP_ENABLED=False'))
    assert audit['training_precision_changed'] and audit['evaluation_precision_changed']


@pytest.mark.parametrize('intervention',['precision','vision_layout'])
def test_evaluation_dtype_change_excludes_time_quality_comparison_and_preserves_nodes(monkeypatch,intervention):
    monkeypatch.setattr(trial_memory,'_pending',{})
    monkeypatch.setattr(trial_memory,'_blocked',{})
    before=sample();after=before.replace('AMP_DTYPE=torch.bfloat16','AMP_DTYPE=torch.float16')
    parent=NS(id='p',code=before,exec_time=10.,metric=NS(value=.8,maximize=True))
    child=NS(id='c',code=after,exec_time=8.,metric=NS(value=.81,maximize=True),is_buggy=False)
    snapshot=copy.deepcopy(child.__dict__)
    trial_memory.remember('own fixture',parent,child,{intervention},'fp16_contiguous')
    blocked,events=trial_memory.observe('own fixture',after)
    assert events[0]['recipe_confounded']
    assert not events[0]['time_or_quality_comparison_eligible_source_screen']
    assert 'evaluation precision changed outside hardware trial' in events[0]['suppression_reasons']
    assert child.__dict__==snapshot and not events[0]['node_modified']


def test_long_alias_chain_and_cycle_terminate_without_runtime_claim():
    aliases=['D0=torch.bfloat16']+[f'D{i}=D{i-1}' for i in range(1,1200)]+['AMP_DTYPE=D1199']
    assert precision_scope(sample('\n'.join(aliases)))['training_dtypes']==['bfloat16']
    assert precision_scope(sample('A=B\nB=A\nAMP_DTYPE=A'))['training_dtypes']==[]
