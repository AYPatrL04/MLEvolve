"""Feedback eligibility replay. Synthetic metrics/times; no training or scoring."""
import copy
from types import SimpleNamespace as NS
import pytest
from hwkg import trial_memory
from tests.test_hwkg_v812 import source, STANDALONE


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.setattr(trial_memory, '_pending', {})
    monkeypatch.setattr(trial_memory, '_blocked', {})


def observe(before, after, *, buggy=False, target='bf16_channels_last', ids=('vision_layout',),
            before_time=10., after_time=12., before_metric=.8, after_metric=.7):
    parent = NS(id='fixture-parent', code=before, exec_time=before_time,
                metric=NS(value=before_metric, maximize=True))
    child = NS(id='fixture-child', code=after, exec_time=after_time, is_buggy=buggy,
               metric=NS(value=after_metric, maximize=True), _term_out=['synthetic fixture'])
    parent_snapshot = copy.deepcopy(parent.__dict__)
    child_snapshot = copy.deepcopy(child.__dict__)
    trial_memory.remember('own scope fixture', parent, child, set(ids), target)
    blocked, events = trial_memory.observe('own scope fixture', before)
    assert parent.__dict__ == parent_snapshot and child.__dict__ == child_snapshot
    return blocked, events


def test_actual_extra_window_work_change_does_not_blame_hardware_profile():
    _, before = source('v88')
    after = before.replace('WIN = 20', 'WIN = 40')
    assert before != after
    blocked, events = observe(before, after)
    event = events[0]
    assert event['recipe_scope']['recipe_syntax_changed'] is False
    assert event['recipe_confounded'] and not event['time_or_quality_comparison_eligible_source_screen']
    assert len(event['work_scope_comparison_exclusions']) == 2
    assert event['suppression_reasons'] == [] and blocked == []
    assert not event['node_modified'] and not event['causal_gain_verified']


@pytest.mark.parametrize('side', ['before', 'after', 'both'])
def test_known_standalone_on_either_side_excludes_comparison(side):
    unknown = STANDALONE.replace('    data_it=iter(loader)\n', '')
    before = STANDALONE if side in ('before', 'both') else unknown
    after = STANDALONE if side in ('after', 'both') else unknown
    blocked, events = observe(before, after)
    assert blocked == [] and not events[0]['time_or_quality_comparison_eligible_source_screen']
    exclusions = events[0]['work_scope_comparison_exclusions']
    assert len(exclusions) == (2 if side == 'both' else 1)
    assert all(s.split(':')[0] in (('before', 'after') if side == 'both' else (side,)) for s in exclusions)
    assert all(not x['runtime_verified'] and not x['epoch_budget_verified']
               for x in events[0]['training_window_scope_source_screen'].values())


def test_normal_embedded_window_length_is_not_extra_training_recipe():
    _, before = source('v89')
    blocked, events = observe(before, before)
    event = events[0]
    assert all(x['status'] == 'embedded_epoch_window_source_screen'
               for x in event['training_window_scope_source_screen'].values())
    assert event['work_scope_comparison_exclusions'] == []
    assert event['time_or_quality_comparison_eligible_source_screen']
    assert not event['recipe_confounded'] and blocked == ['bf16_channels_last']
    assert len(event['suppression_reasons']) == 2 and not event['causal_gain_verified']


@pytest.mark.parametrize('source_code', [
    'x=1',
    STANDALONE.replace('    data_it=iter(loader)\n', ''),
    STANDALONE.replace('    data_it=iter(loader)\n', '    data_it=iter(data_it)\n'),
])
def test_unknown_source_not_blanket_standalone_exclusion(source_code):
    blocked, events = observe(source_code, source_code)
    assert events[0]['work_scope_comparison_exclusions'] == []
    assert events[0]['time_or_quality_comparison_eligible_source_screen']
    assert all(x['status'] == 'unknown' for x in events[0]['training_window_scope_source_screen'].values())
    assert blocked == ['bf16_channels_last']
    assert not events[0]['causal_gain_verified']


def test_failure_warning_retained_independently_of_unmatched_work_scope():
    blocked, events = observe(STANDALONE, STANDALONE, buggy=True)
    assert not events[0]['time_or_quality_comparison_eligible_source_screen']
    assert events[0]['suppression_reasons'] == ['native candidate failed; attribution unknown']
    assert blocked == ['bf16_channels_last']


def test_eval_precision_warning_retained_independently_of_unmatched_scope():
    _, before = source('v88')
    after = before.replace('AMP_DTYPE = torch.bfloat16', 'AMP_DTYPE = torch.float16')
    blocked, events = observe(before, after, ids=('precision',), target='fp16_channels_last')
    assert events[0]['recipe_scope']['evaluation_precision_changed']
    assert events[0]['suppression_reasons'] == ['evaluation precision changed outside hardware trial']
    assert blocked == ['fp16_channels_last']


def test_inferred_target_not_suppressed_by_scope_alone():
    blocked, events = observe(STANDALONE, STANDALONE, target=None)
    assert events[0]['target_profile'] == 'bf16_contiguous'
    assert blocked == [] and events[0]['suppression_reasons'] == []


def test_positive_fake_metric_does_not_certify_gain_with_standalone_work():
    blocked, events = observe(STANDALONE, STANDALONE, after_time=8., after_metric=.9)
    assert blocked == [] and not events[0]['time_or_quality_comparison_eligible_source_screen']
    assert not events[0]['causal_gain_verified']


def test_not_yet_executed_nodes_remain_pending():
    blocked, events = observe(STANDALONE, STANDALONE, after_time=None)
    assert events == [] and blocked == [] and len(trial_memory._pending) == 1


def test_work_exclusion_consumed_once_without_changing_native_state():
    blocked, events = observe(STANDALONE, STANDALONE)
    assert len(events) == 1 and not trial_memory._pending
    assert trial_memory.observe('own scope fixture', STANDALONE) == ([], [])
    assert not trial_memory._blocked
