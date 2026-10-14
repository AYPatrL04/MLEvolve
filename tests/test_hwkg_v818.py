"""Explicit contract reminders and conservative inference-work feedback scope."""
from pathlib import Path
import copy
import pytest
from hwkg.contracts import scope_reminder
from hwkg.exploration_policy import guidance
from hwkg.native_evidence import recipe_changes
from tests.test_hwkg_v814 import EDGES,RUNTIME,GPU
from tests.test_hwkg_v816 import GROUPED
from tests.test_hwkg_v8 import build

TASK='Offline-only: no network downloads. Fit train.csv only; validation.csv is eval-only.'
R={**RUNTIME,'adamw_fused_api':True,'grad_scaler_api':True}


@pytest.mark.parametrize('prompt',[TASK,{'system':'base','user':TASK,'assistant':'preview'}])
def test_actual_provider_prompt_contract_survives_cleaned_task(prompt):
    frozen=copy.deepcopy(prompt)
    note=scope_reminder('cleaned task',prompt)
    assert 'offline flags/local weights' in note and 'train.csv only' in note
    assert prompt==frozen


def test_no_invented_offline_or_validation_protocol():
    assert scope_reminder('GPU classification')==''
    assert scope_reminder('No network downloads')=='Keep offline flags; use local weights only.'
    assert scope_reminder('Fit train.csv only; validation.csv is eval-only.')=='Fit train.csv only; validation is eval-only.'


@pytest.mark.parametrize('identity',[False,True])
def test_draft_keeps_hardware_choices_and_bounded_contract(identity):
    note=scope_reminder(TASK)
    text,audit=guidance('draft','',GPU,EDGES,R,phase='plan_and_code',
        parent_execution_succeeded=None,identity_already_present=identity,contract_note=note)
    assert len(text)<=340 and note in text
    assert 'BF16' in text and 'fused=True' in text and 'GradScaler' in text
    assert 'same budget/outputs' in text and 'no extra passes' in text
    assert audit['host_header_chars']+audit['advice_body_chars']==len(text)
    assert not audit['measurements_verified']


@pytest.mark.parametrize('phase',['planner','coder','plan_and_code'])
def test_parent_reminder_is_advice_and_keeps_one_hardware_change(phase):
    text,audit=build(code=GROUPED,runtime=R,phase=phase,failure_text='',provider_prompt=TASK)
    assert scope_reminder(TASK) in text and len(text)<=750
    assert audit['advice_ids']==['optimizer_fused_adamw']
    assert audit['contract_runtime_enforced'] is False and audit['added_provider_calls']==0


def test_real_Cassava_TTA_drift_now_excludes_comparison_without_changing_source():
    root=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkd-v817-cassava-public-s42-15plus15-20261014T053000Z/treatment/candidates'
    a=(root/'63dbd6e497c84dfdb5aaf836b980ab17/candidate.py').read_text()
    b=(root/'71113a872d9f47b5a8136014aa401658/candidate.py').read_text()
    row=recipe_changes(a,b)
    assert row['evaluation_work_syntax_changed'] and row['recipe_syntax_changed']
    assert row['semantic_preservation_certified'] is False


def test_training_only_edit_does_not_change_named_inference_work():
    a='x=1\ndef predict(x):\n return model(x)\n'
    b=a.replace('x=1','x=2')
    assert recipe_changes(a,b)['evaluation_work_syntax_changed'] is False


def test_unknown_names_do_not_certify_no_dynamic_TTA_change():
    a='def arbitrary(x):\n return x\n'
    row=recipe_changes(a,a)
    assert row['evaluation_work_syntax_changed'] is False
    assert row['semantic_preservation_certified'] is False
