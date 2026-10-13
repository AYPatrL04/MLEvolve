import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from hwkg.routing import hardware_context, hardware_phase, with_hardware_prompt
from hwkg.staged_policy import guidance_v5, source_adoption

ROOT=Path(__file__).resolve().parents[1]
GPU=NS(name='NVIDIA GeForce RTX 5080',major=12,minor=0,total_memory=16*1024**3)
EDGES={k:{'properties':{'verified':True,'support_level':'supported'}} for k in ['feat:amp','feat:bf16']}
CFG=NS(agent=NS(code=NS(model='deepseek-flash')))


@pytest.fixture
def enabled(tmp_path,monkeypatch):
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH',str(ROOT/'hwkg/graph.json'))
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_SHA256','0d422cfe81e2531980204e71b49fa9e5ffed9c2606a9a5569e7a43ebfc0078b4')
    monkeypatch.setenv('MLEVOLVE_HWKG_HARDWARE_ID','nvidia.blackwell.geforce_rtx_5080.spec')
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v5')
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR',str(tmp_path))
    monkeypatch.delenv('MLEVOLVE_HWKG_EXPLORATORY',raising=False)
    with patch('hwkg.injection._visible_cuda_device',return_value=GPU):yield tmp_path


@pytest.mark.parametrize('phase',['planner','coder','merge','plan_and_code'])
def test_phase_budget_and_numerical_safety(phase):
    text,audit=guidance_v5('debug','tweet text','import torch',GPU,EDGES,phase=phase,failure_text='epoch loss nan')
    assert len(text)<=1600
    assert 'BF16 is not a NaN fix' in text and 'do not skip bad batches' in text
    if phase!='planner':assert 'after the first normal optimizer step' in text and 'actual parameter dtype' in text
    assert audit['source_observations']['runtime_verified'] is False


def test_planner_to_coder_refresh_catches_cpu_to_transformer(enabled):
    sent=[]
    @with_hardware_prompt
    def send(p,cfg):sent.append(p);return 'response'
    @hardware_phase('planner')
    def plan():
        send('plan',CFG)
        return {'plan':{'model_design':'Use a RoBERTa transformer on CUDA'}}
    @hardware_phase('coder')
    def code():send('implement planned model',CFG)
    @hardware_context('improve')
    def run(agent,parent_node):
        plan();code()
        return NS(id='new',code='import torch\nx=torch.isfinite(loss)')
    run(NS(task_desc='tweet text'),NS(id='parent',code='model = LogisticRegression()',_term_out=[]))
    assert '[plan]' in sent[0] and '[precision]' not in sent[0]
    assert '[precision]' in sent[1] and '[padding]' in sent[1]
    assert 'torch.cuda.is_bf16_supported()' in sent[1]
    context_rows=[json.loads(s) for s in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    assert set(context_rows[-1]['evidence_sources'])=={'precision','padding'}
    rows=[json.loads(s) for s in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    dispatches=[r for r in rows if r['event']=='provider_dispatch']
    assert [r['phase'] for r in dispatches]==['planner','coder']
    assert len({r['context_sha256'] for r in dispatches})==2
    ret=next(r for r in rows if r['event']=='candidate_return')
    assert ret['source_observations']['finite_check_call_present']
    assert not ret['source_observations']['runtime_verified']
    send('outside',CFG);assert sent[-1]=='outside'


def test_phase_reset_after_exception_and_noop_without_candidate(enabled):
    from hwkg.routing import _phase
    @hardware_phase('merge')
    def merge():raise RuntimeError('merge failed')
    @hardware_context('draft')
    def run(agent):
        with pytest.raises(RuntimeError):merge()
        assert _phase.get()=='plan_and_code'
    run(NS(task_desc='text'))
    with pytest.raises(RuntimeError):merge()
    assert _phase.get()=='plan_and_code'


def test_comments_and_literal_dtype_claims_are_not_source_proof():
    evidence=source_adoption('# torch.isfinite(params)\nprint("dtype=fp32 autocast()")')
    assert not evidence['finite_check_call_present'] and not evidence['dtype_read_present']
    assert not evidence['autocast_call_present'] and not evidence['runtime_verified']


def test_v4_phase_hooks_do_not_change_existing_context(enabled,monkeypatch):
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v4')
    seen=[]
    @with_hardware_prompt
    def send(p,cfg):seen.append(p)
    @hardware_phase('planner')
    def plan():send('same',CFG);return 'roberta'
    @hardware_phase('coder')
    def code():send('same',CFG)
    @hardware_context('improve')
    def run(agent,parent_node):plan();code()
    run(NS(task_desc='text'),NS(id='p',code='x = 1',_term_out=[]))
    assert seen[0]==seen[1] and 'HWKG v4' in seen[0]


@pytest.mark.parametrize('path',['agents/planner/base_planner.py','agents/planner/planner_with_memory.py',
    'agents/coder/base_coder.py','agents/coder/stepwise_coder.py','agents/coder/diff_coder/diff_generate.py'])
def test_new_hook_files_have_identical_upstream_logic(path):
    original=subprocess.check_output(['git','show','9c5c8a3b23f0361708b59a401452dddc00f97189:'+path],cwd=ROOT,text=True)
    tree=ast.parse((ROOT/path).read_text())
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and (n.module or '').startswith('hwkg'))]
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
            n.decorator_list=[d for d in n.decorator_list if not(isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='hardware_phase')]
    assert ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False)
