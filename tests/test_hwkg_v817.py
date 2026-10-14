"""Separate shared task goals from initial, explicitly counted host context."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import pytest
from hwkg.exploration_policy import guidance
from hwkg.routing import hardware_context, with_hardware_prompt
from tests.test_hwkg_v814 import EDGES, RUNTIME, GPU
from tests.test_hwkg_v8 import enabled
from tests.test_hwkg_v6 import CFG

ROOT=Path(__file__).resolve().parents[1]
TASK=(ROOT/'hwkg/tasks/CIFAR100_NEUTRAL_TASK_V1.md').read_text()
R={**RUNTIME,'adamw_fused_api':True,'grad_scaler_api':True}


def draft(**kw):
    return guidance('draft','',kw.pop('device',GPU),EDGES,kw.pop('runtime',R),
        phase='plan_and_code',parent_execution_succeeded=None,**kw)


def test_neutral_task_keeps_public_goals_and_budget_without_host_or_tuning_presets():
    for token in ['40000 rows','10000 rows','128x128','six epochs','360 seconds',
                  '31GiB','validation_probabilities.npy','ResNet50_Weights.IMAGENET1K_V2']:
        assert token in TASK
    for token in ['5080','nvidia','geforce','capability 12','cu130','2.14.1','bf16',
                  'fp16','fused=true','channels_last','cuda']:
        assert token not in TASK.casefold()
    assert 'custom public CIFAR-100' in TASK and 'not a verbatim Kaggle' in TASK
    assert 'Runtime discovery' in TASK


def test_initial_context_contains_actual_host_build_and_memory_not_task_literals():
    text,audit=draft()
    assert len(text)<=340
    assert GPU.name in text and 'CUDA13.0' in text and 'Torch2.14.1+cu130' in text
    assert f'VRAMtotal{GPU.total_memory//1024**2}MiB' in text
    assert '[precision_choice]' in text and '[optimizer_choice]' in text
    assert audit['host_header_chars']+audit['advice_body_chars']==len(text)
    assert audit['hardware_fact_values']=={
        'compute_capability':'12.0','backend':'cuda','hip_runtime_version':None,
        'gpu_name':GPU.name,'cuda_runtime_version':'13.0',
        'torch_version':'2.14.1+cu130','total_vram_mib':GPU.total_memory//1024**2}
    assert audit['host_facts_and_advice_counted_in_total']
    assert audit['calibration_trial'] is None and not audit['measurements_verified']


def test_runtime_and_device_values_are_read_instead_of_embedded_workstation_constants():
    device=NS(**{**vars(GPU),'total_memory':15*1024**3})
    text,audit=draft(device=device,runtime={**R,'torch':'test-build','cuda':'test-runtime'})
    assert 'Torchtest-build' in text and 'CUDAtest-runtime' in text and '15360MiB' in text
    assert audit['hardware_fact_values']['total_vram_mib']==15360


def test_existing_identity_deduplicates_facts_without_hiding_extra_context_count():
    text,audit=draft(identity_already_present=True)
    assert 'VRAMtotal' not in text and GPU.name not in text
    assert audit['hardware_fact_values']=={'compute_capability':'12.0','backend':'cuda'}
    assert audit['host_header_chars']+audit['advice_body_chars']==len(text)
    assert audit['draft_identity_facts_deduplicated']


@pytest.mark.parametrize('kw',[{'max_chars':40}])
def test_unsupported_or_overbudget_context_stays_absent(kw):
    assert draft(**kw) is None


def test_no_install_reconfiguration_fake_rate_or_extra_measurement_instruction():
    text,_=draft()
    for forbidden in ['TFLOPS','speedup','pip install','set_default_dtype','extra benchmark']:
        assert forbidden not in text
    assert 'Unmeasured speed/quality' in text and 'no extra passes' in text


def test_only_treatment_provider_gets_hardware_context_and_native_arguments_unchanged(enabled,monkeypatch):
    monkeypatch.setattr('hwkg.host_policy.runtime_facts',lambda:R)
    prompt={'system':'original role','user':TASK+'\n# Response format\noriginal fences',
            'assistant':'dataset files: train/validation/test'}
    original=copy.deepcopy(prompt);calls=[]
    # The reference path is the unchanged native prompt, with no component wrapper.
    def native(p,cfg,**kwargs):calls.append((copy.deepcopy(p),cfg,kwargs));return 'native response'
    native(prompt,CFG,max_tokens=16384,temperature=1.)
    provider=with_hardware_prompt(native)
    @hardware_context('draft')
    def treatment(agent):return provider(prompt,CFG,max_tokens=16384,temperature=1.)
    assert treatment(NS(task_desc=TASK))=='native response'
    assert calls[0][0]==original==prompt
    assert calls[1][0]['system']==original['system'] and calls[1][0]['assistant']==original['assistant']
    assert GPU.name in calls[1][0]['user'] and GPU.name not in calls[0][0]['user']
    assert all(c[1] is CFG and c[2]=={'max_tokens':16384,'temperature':1.} for c in calls)
    audit=json.loads((enabled/'prompt-audit.jsonl').read_text())
    assert audit['chars']==audit['host_header_chars']+audit['advice_body_chars']
    assert audit['added_provider_calls']==0


def test_original_native_draft_and_raw_data_preview_do_not_reintroduce_task_hardware(tmp_path):
    """Invoke original prompt assembly in a fresh process with only provider stubs."""
    (tmp_path/'description.md').write_text(TASK)
    (tmp_path/'train.csv').write_text('id,image_row,target\n0,0,0\n1,1,1\n')
    upstream=ROOT.parent/'MLEvolve/records/hwkg-v815-native-5080-cifar100-local-s42-15plus15-rerun-20261014T013057Z/source-baseline'
    script='''import json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,sys.argv[1])
from agents import draft_agent as d
from engine.search_node import SearchNode
from utils.data_preview import generate
data=Path(sys.argv[2]);captured=[]
def provider(agent,prompt):captured.append(prompt);return 'offline fixture','pass'
d.plan_and_code_query=provider
d.register_node=lambda *a,**k:None
agent=NS(task_desc=(data/'description.md').read_text(),data_preview=generate(data),
 virtual_root=SearchNode(code='',stage='root'),use_coldstart=False,
 use_stepwise_generation=False,current_step=0,start_time=time.time(),
 cfg=NS(exec=NS(timeout=360),pretrain_model_dir=''),
 acfg=NS(code=NS(model='deepseek-flash'),time_limit=10800,steps=15))
d.run(agent)
print(json.dumps(captured[0],ensure_ascii=False))
'''
    p=subprocess.run([sys.executable,'-c',script,str(upstream),str(tmp_path)],
                     capture_output=True,text=True,check=True,timeout=20)
    prompt=json.loads(p.stdout);actual=json.dumps(prompt).casefold()
    for fact in ['5080','sm_120','capability 12.0','cu130','2.14.1','bf16','fused=true']:
        assert fact not in actual
    assert 'CIFAR-100' in prompt['user'] and 'description.md' in prompt['assistant']
