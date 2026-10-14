"""Use the saved native provider prompt, after cleaner and data preview."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg.routing import hardware_context, with_hardware_prompt
from tests.test_hwkg_v8 import enabled, TASK
from tests.test_hwkg_v6 import CFG


def actual_native_prompt():
    root=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkg-v87-native-5080-cifar100-local-s42-light4-20261013T142230Z'
    if not root.exists():pytest.skip('public native prompt fixture unavailable')
    node=json.loads((root/'treatment/candidates/c2edcdd9ad564bf3b85b6c857b23c35f/node.json').read_text())
    prompt=json.loads(node['prompt_input'])
    dispatch=[json.loads(l) for l in (root/'treatment/hwkg-audit/dispatch-audit.jsonl').read_text().splitlines()]
    original=next(r for r in dispatch if r['event']=='provider_dispatch')
    assert hashlib.sha256(json.dumps(prompt,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==original['base_prompt_sha256']
    return prompt


def test_actual_provider_identity_deduplicates_facts_but_keeps_hypotheses(enabled):
    prompt=actual_native_prompt();frozen=copy.deepcopy(prompt);sent=[]
    @with_hardware_prompt
    def provider(p,cfg,**kwargs):sent.append((p,cfg,kwargs));return 'native unchanged'
    @hardware_context('draft')
    def draft(agent):return provider(prompt,CFG,max_tokens=16384),provider(prompt,CFG,max_tokens=16384)
    assert draft(NS(task_desc='Cleaned CIFAR100 PyTorch CUDA task; host facts removed'))==('native unchanged',)*2
    assert prompt==frozen and sent[0]==sent[1]
    assert sent[0][1] is CFG and sent[0][2]=={'max_tokens':16384}
    assert '[precision_choice]' in sent[0][0]['user']
    rows=[json.loads(l) for l in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]['chars']>0 and rows[0]['draft_identity_facts_deduplicated']
    assert rows[0]['draft_identity_check_basis']=='actual provider prompt'
    dispatch=[json.loads(l) for l in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert sum(r['event']=='provider_dispatch' for r in dispatch)==2


def test_parentless_cache_distinguishes_different_provider_identity_context(enabled):
    full=actual_native_prompt();sent=[]
    @with_hardware_prompt
    def provider(prompt,cfg):sent.append(prompt);return 'native answer'
    @hardware_context('draft')
    def draft(agent):
        provider('Original output format',CFG)
        provider(full,CFG)
        provider('Original output format',CFG)
    draft(NS(task_desc=TASK))
    assert 'HWKG v8.17' in sent[0] and '[precision_choice]' in sent[1]['user'] and sent[2]==sent[0]
    rows=[json.loads(l) for l in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(r['chars']>0 for r in rows)


def test_only_own_block_replaced_when_prompt_gains_facts_but_pattern_remains(enabled):
    full=actual_native_prompt();sent=[]
    facts='\n'.join(v for k,v in full.items() if k in ('system','user','assistant') and isinstance(v,str))
    @with_hardware_prompt
    def provider(p,cfg):sent.append(p);return 'native answer'
    @hardware_context('draft')
    def draft(agent):
        provider({'system':'system','user':'Original format','assistant':'prefill'},CFG)
        augmented={**sent[0],'user':sent[0]['user']+'\n'+facts}
        provider(augmented,CFG)
    draft(NS(task_desc=TASK))
    assert '[precision_choice]' in sent[1]['user']
    assert sent[1]['user'].count('HWKG v8.17')==1
    rows=[json.loads(l) for l in (enabled/'prompt-audit.jsonl').read_text().splitlines()]
    block='\n# Hardware knowledge (advisory)\n'+rows[-1]['context_text']+'\n\n'
    assert sent[1]['user'].replace(block,'')=='Original format\n'+facts
    assert sent[1]['system']=='system' and sent[1]['assistant']=='prefill'


def test_own_card_not_treated_as_native_identity_on_reused_prompt(enabled):
    sent=[]
    @with_hardware_prompt
    def provider(p,cfg):sent.append(p);return 'native answer'
    @hardware_context('draft')
    def draft(agent):provider('Task and original format',CFG);provider(sent[0],CFG)
    draft(NS(task_desc=TASK))
    assert sent[0]==sent[1] and sent[1].count('HWKG v8.17')==1
