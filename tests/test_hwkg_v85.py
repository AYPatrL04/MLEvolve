"""Optional hardware injection recovery, not a change to native API retries."""
import json
from types import SimpleNamespace as NS
import pytest
from hwkg.routing import hardware_context, hardware_phase, with_hardware_prompt, _append_feedback_advice
from tests.test_hwkg_v8 import enabled, build, CODE, TASK, log
from tests.test_hwkg_v6 import CFG


def test_parentless_draft_has_compact_capability_hypotheses():
    text,audit=build(stage='draft',code='',parent_execution_succeeded=None,failure_text='')
    assert text and len(text)<=340
    assert 'sm_120' in text and 'NVIDIA GeForce RTX 5080' in text
    assert 'BF16' in text and 'FP16 needs GradScaler' in text
    assert 'Unmeasured speed/quality' in text and 'torch.autocast' not in text
    assert audit['advice_ids']==['precision_choice'] and audit['draft_recipe_forced'] is False
    assert audit['calibration_trial'] is None and not audit['measurements_verified']


@pytest.mark.parametrize('empty',['','  \n'])
def test_empty_response_withdraws_only_hardware_content_on_original_retry(enabled,empty):
    sent=[];responses=[empty,'brief outline\n```python\nx=1\n```']
    original='Native task\n# Response format\nOriginal exact fences.'
    @with_hardware_prompt
    def provider(prompt,cfg,**kwargs):
        sent.append((prompt,cfg,kwargs));return responses[len(sent)-1]
    @hardware_context('draft')
    def run(agent):
        first=provider(original,CFG,temperature=1.,max_tokens=16384)
        second=provider(original,CFG,temperature=1.,max_tokens=16384)
        return first,second
    assert run(NS(task_desc=TASK))==tuple(responses)
    assert len(sent)==2 and 'HWKG v8.18' in sent[0][0] and sent[1][0]==original
    assert sent[0][1] is CFG and sent[1][1] is CFG and sent[0][2]==sent[1][2]
    assert sent[0][0].endswith('# Response format\nOriginal exact fences.')
    rows=[json.loads(s) for s in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert sum(s['event']=='empty_response_advice_withdrawn' for s in rows)==1
    assert sum(s['event']=='advice_suppressed_after_empty' for s in rows)==1


def test_withdrawal_is_candidate_local_and_removes_only_own_reused_block(enabled):
    original={'system':'original system','user':'User task\n# Response format\nOriginal format','assistant':'prefill'}
    sent=[]
    @with_hardware_prompt
    def provider(prompt,cfg):sent.append(prompt);return '' if len(sent)==1 else 'native unchanged'
    @hardware_context('draft')
    def run(agent):
        provider(original,CFG)
        delivered=sent[-1]
        assert provider(delivered,CFG)=='native unchanged'
    run(NS(task_desc=TASK))
    assert sent[1]==original and original['assistant']=='prefill'
    @hardware_context('draft')
    def next_candidate(agent):return provider(original,CFG)
    next_candidate(NS(task_desc=TASK));assert 'HWKG v8.18' in sent[-1]['user']
    assert 'HWKG' not in original['user']


def test_nonempty_format_failure_does_not_change_parser_or_response(enabled):
    calls=[]
    @with_hardware_prompt
    def provider(prompt,cfg):calls.append(prompt);return 'not Python; native parser decides'
    @hardware_context('draft')
    def run(agent):return provider('original',CFG),provider('original',CFG)
    assert run(NS(task_desc=TASK))==('not Python; native parser decides',)*2
    assert len(calls)==2 and all('HWKG' in s for s in calls)


def test_unrelated_hardware_section_is_not_removed(enabled):
    original='Task\n# Hardware knowledge (advisory)\nuser-owned text\n\n# Response format\nOriginal.'
    calls=[]
    @with_hardware_prompt
    def provider(prompt,cfg):calls.append(prompt);return ''
    @hardware_context('draft')
    def run(agent):provider(original,CFG);provider(calls[0],CFG)
    run(NS(task_desc=TASK));assert calls[1]==original
