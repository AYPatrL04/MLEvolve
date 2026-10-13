import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from hwkg.host_policy import guidance_v6, workload
from hwkg.injection import hardware_guidance
from hwkg.routing import hardware_context, hardware_phase, with_hardware_prompt

ROOT=Path(__file__).resolve().parents[1]
GRAPH=json.loads((ROOT/'hwkg/graph.json').read_text())
FEATURES={n['id']:n for n in GRAPH['nodes'] if n['id'].startswith('feat:')}
DEVICES={
    '5080':NS(name='NVIDIA GeForce RTX 5080',major=12,minor=0,total_memory=16*1024**3),
    'a10':NS(name='NVIDIA A10',major=8,minor=6,total_memory=23*1024**3),
}
IDS={'5080':'hw:nvidia.blackwell.geforce_rtx_5080.spec','a10':'hw:nvidia.ampere.a10.spec'}
RUNTIME={'torch':'test','cuda':'13.0','arch_list':['sm_86','sm_120'],'bf16_supported':True,
         'sdpa_api':True,'flash_sdp_built':True,'free_mib_snapshot':14000,
         'transformer_engine_package':False,'torchao_package':False}
CFG=NS(agent=NS(code=NS(model='deepseek-flash')))


def build(host='5080',stage='improve',task='tweet text classification',code='import torch\nmodel=AutoModel.from_pretrained("roberta")',**kw):
    node=copy.deepcopy(next(n for n in GRAPH['nodes'] if n['id']==IDS[host]))
    edges={e['to']:copy.deepcopy(e) for e in GRAPH['edges'] if e['from']==IDS[host]}
    runtime={**RUNTIME,**kw.pop('runtime',{})}
    if 'hardware' in kw:node=kw.pop('hardware')
    if 'edges' in kw:edges=kw.pop('edges')
    return guidance_v6(stage,task,code,DEVICES[host],node,FEATURES,edges,runtime,**kw)


@pytest.fixture
def enabled(tmp_path,monkeypatch):
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH',str(ROOT/'hwkg/graph.json'))
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_SHA256','3842eff0a7742c42b3777a01a6f58cfbec42b37c442d9bb9f5b080321ff7ed62')
    monkeypatch.setenv('MLEVOLVE_HWKG_HARDWARE_ID',IDS['5080'].removeprefix('hw:'))
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR',str(tmp_path))
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v6')
    monkeypatch.delenv('MLEVOLVE_HWKG_OPTIMIZATION_MODE',raising=False)
    with patch('hwkg.injection._visible_cuda_device',return_value=DEVICES['5080']),patch('hwkg.host_policy.runtime_facts',return_value=RUNTIME):yield tmp_path


def test_explicit_v6_reaches_host_graph_decisions_and_restores_tf32(enabled):
    text=hardware_guidance('draft','tweet text classification')
    row=json.loads((enabled/'prompt-audit.jsonl').read_text())
    assert row['policy']=='hardware-v6' and 'sm_120' in text
    assert 'BF16-autocast' in text and "set_float32_matmul_precision('high')" in text
    assert 'feat:tf32' in row['selected_feature_ids']
    assert 'memory' in row['advice_ids']
    assert any(p['graph_patterns'] for p in row['decision_provenance'])
    assert row['rejected_features']['feat:nvfp4']=='required backend not installed'
    assert len(text)<=1600


def test_host_counterfactual_changes_capacity_target_and_low_precision_options():
    task='long language model pretraining on a billion token corpus'
    runtime={'transformer_engine_package':True,'torchao_package':True}
    b,ba=build(task=task,runtime=runtime,optimization_mode='aggressive')
    a,aa=build('a10',task=task,runtime=runtime,optimization_mode='aggressive')
    assert 'sm_120' in b and 'sm_86' in a
    assert '16384 MiB' in b and '23552 MiB' in a
    assert any(fid in ba['selected_feature_ids'] for fid in ['feat:nvfp4','feat:mxfp8','feat:fp8'])
    assert not any(fid in aa['selected_feature_ids'] for fid in ['feat:nvfp4','feat:mxfp8','feat:fp8'])
    assert b!=a


def test_graph_and_runtime_both_gate_precision_decisions():
    edges={e['to']:copy.deepcopy(e) for e in GRAPH['edges'] if e['from']==IDS['5080']}
    edges.pop('feat:tf32');edges['feat:bf16']['properties']['verified']=False
    text,audit=build(edges=edges)
    assert '[precision]' not in text
    assert 'feat:bf16' not in audit['selected_feature_ids']
    text,audit=build(runtime={'bf16_supported':False})
    assert 'BF16-autocast' not in text and 'TF32-capable' in text


def test_host_patterns_are_retrieved_instead_of_unconditional_generic_padding():
    node=copy.deepcopy(next(n for n in GRAPH['nodes'] if n['id']==IDS['5080']))
    node['properties']['recommended_patterns']=[]
    text,audit=build(hardware=node)
    assert '[batching]' not in text and '[memory]' not in text
    assert 'attention' in audit['advice_ids']


def test_cpu_candidate_spends_no_injection_tokens(enabled):
    sent=[]
    @with_hardware_prompt
    def send(p,cfg):sent.append(p);return 'response'
    @hardware_context('improve')
    def run(agent,parent_node):return send('base prompt',CFG)
    run(NS(task_desc='tabular regression'),NS(id='p',code='model=RandomForestRegressor()',_term_out=[]))
    assert sent==['base prompt']
    assert not (enabled/'dispatch-audit.jsonl').exists()
    row=json.loads((enabled/'prompt-audit.jsonl').read_text())
    assert row['chars']==0 and row['context_text'] is None


def test_default_phase_refreshes_parent_cpu_to_neural_plan(enabled):
    sent=[]
    @with_hardware_prompt
    def send(p,cfg):sent.append(p);return 'response'
    @hardware_phase('planner')
    def plan():send('plan',CFG);return {'plan':{'model_design':'Use RoBERTa on CUDA'}}
    @hardware_phase('coder')
    def code():send('code',CFG)
    @hardware_context('improve')
    def run(agent,parent_node):plan();code();return NS(id='new',code='import torch')
    run(NS(task_desc='tweet text'),NS(id='old',code='model=LogisticRegression()',_term_out=[]))
    assert '[plan]' in sent[0] and '[attention]' in sent[1]
    rows=[json.loads(s) for s in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert [x['phase'] for x in rows if x['event']=='provider_dispatch']==['planner','coder']
    assert any(x.get('source_observations',{}).get('runtime_verified') is False for x in rows)


def test_cpu_coder_skip_keeps_prior_planner_trace_link(enabled):
    @with_hardware_prompt
    def send(p,cfg):return 'response'
    @hardware_phase('planner')
    def plan():send('plan',CFG);return 'Keep LogisticRegression on CPU'
    @hardware_phase('coder')
    def code():send('CPU code',CFG)
    @hardware_context('improve')
    def run(agent,parent_node):
        plan();code();return NS(id='cpu-child',code='model=LogisticRegression()')
    run(NS(task_desc='tabular regression'),NS(id='p',code='model=LogisticRegression()',_term_out=[]))
    rows=[json.loads(s) for s in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert len([x for x in rows if x['event']=='provider_dispatch'])==1
    returned=next(x for x in rows if x['event']=='candidate_return')
    assert returned['candidate_id']=='cpu-child'
    assert returned['trace_id']==rows[0]['trace_id']


def test_debug_defers_tuning_but_keeps_host_numeric_choices():
    text,audit=build(stage='debug',failure_text='NaN after optimizer step')
    assert '[diagnose]' in text and '[precision]' in text
    assert 'BF16 is not a NaN repair' in text
    assert '[batching]' not in text and '[attention]' not in text


def test_kernel_limits_only_for_actual_custom_kernel_candidate():
    text,audit=build(code='import triton\nimport torch')
    assert '99 KiB/block' in text and '48 resident warps/SM' in text
    assert 'feat:sm_120' in audit['selected_feature_ids']
    text,_=build()
    assert '99 KiB' not in text


def test_low_precision_trials_need_backend_long_workload_and_mode():
    runtime={'transformer_engine_package':True,'torchao_package':True}
    text,audit=build(runtime=runtime,optimization_mode='aggressive')
    assert '[nvfp4]' not in text and 'workload' in audit['rejected_features']['feat:nvfp4']
    text,audit=build(task='long language model pretraining corpus',runtime=runtime)
    assert '[nvfp4]' not in text and 'aggressive' in audit['rejected_features']['feat:nvfp4']
    assert not workload('tweet classifier','model=AutoModel.from_pretrained("roberta")','')['long_lm']
    with pytest.raises(ValueError):build(optimization_mode='unknown')


def test_stage_and_workload_render_fit_budgets():
    for stage in ['draft','improve','debug','fusion','evolution','aggregation']:
        for phase in ['planner','coder','merge','plan_and_code']:
            for task in ['tweet text classification','CNN image classification','long language model pretraining corpus']:
                text,audit=build(stage=stage,phase=phase,task=task,code='' if stage=='draft' else 'import torch')
                assert text and len(text)<=1600
                assert 'Preserve task metric/split/seed/threshold/budget' in text
                assert audit['decision_provenance']
