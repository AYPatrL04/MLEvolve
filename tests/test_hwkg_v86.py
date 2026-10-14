"""Component-only regressions for observed EMA confounding and native NCHW logs."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg.native_evidence import recipe_changes, auxiliary_recipe_signature
from hwkg.trial_memory import remember, observe, _pending, _blocked
from tests.test_hwkg_v8 import CODE, TASK, build, read, log


def historical_pair():
    root = Path(__file__).resolve().parents[2] / 'MLEvolve/records/hwkg-v85-native-5080-cifar100-local-s42-4plus4-20261013T095159Z/treatment/candidates'
    if not root.exists():
        pytest.skip('historical public regression fixture unavailable')
    return [(root / candidate / 'candidate.py').read_text() for candidate in
            ['e69c2c626f93433697860a96f6ff8124', '93f62d1526ec4aafbaee7f753c6f7ead']]


def test_actual_ema_checkpoint_addition_is_confounded():
    before, after = historical_pair()
    row = recipe_changes(before, after)
    assert row['averaging_recipe_syntax_changed']
    assert row['checkpoint_selection_syntax_changed']
    assert row['recipe_syntax_changed'] and not row['semantic_preservation_certified']
    assert 'not complete semantic' in row['recipe_audit_coverage']


def test_pure_layout_and_identical_ema_remain_source_screen_equal():
    before = CODE + '\nema=ModelEMA(model, decay=.999)\nema.update(model)\n'
    row = recipe_changes(before, before + '\nmodel.to(memory_format=torch.channels_last)')
    assert not row['recipe_syntax_changed'] and not row['averaging_recipe_syntax_changed']
    assert not row['semantic_preservation_certified']
    assert recipe_changes(before, before.replace('decay=.999', 'decay=.99'))['averaging_recipe_syntax_changed']


def test_alias_constructor_bound_updates_and_decay_constants_are_covered():
    code = CODE + '\nfrom torch.optim.swa_utils import AveragedModel as Tracker\nema_decay=.99\naverage=Tracker(model)\naverage.update_parameters(model)'
    assert recipe_changes(code, code.replace('ema_decay=.99', 'ema_decay=.9'))['averaging_recipe_syntax_changed']
    assert recipe_changes(code, code.replace('average.update_parameters(model)', 'average.update_parameters(other)'))['averaging_recipe_syntax_changed']
    assert auxiliary_recipe_signature('SCHEMA = 1\nSCHEMA()')['averaging'] == []


@pytest.mark.parametrize('extra', ['\nmodel.load_state_dict(selected)', '\ntorch.save(model.state_dict(), path)', '\nema.copy_to(model)'])
def test_checkpoint_change_is_separate_from_precision(extra):
    row = recipe_changes(CODE, CODE + extra)
    assert row['checkpoint_selection_syntax_changed'] and row['recipe_syntax_changed']
    assert not row['training_precision_changed'] and not row['evaluation_precision_changed']


@pytest.mark.parametrize('before,after', [('', CODE), (CODE, 'bad('), ('bad(', CODE)])
def test_missing_or_invalid_recipe_remains_unknown(before, after):
    row = recipe_changes(before, after)
    assert row['recipe_syntax_changed'] is None
    assert row['averaging_recipe_syntax_changed'] is None
    assert not row['semantic_preservation_certified']


@pytest.mark.parametrize('layout', ['channels_first (contiguous NCHW)', 'channels_first (NCHW)'])
def test_observed_descriptive_nchw_is_allowlisted_but_not_verified(layout):
    row = read(log(model_memory_format=layout, input_memory_format=layout))
    assert row['window']['model_layout'] == 'contiguous'
    assert row['window']['input_layout'] == 'contiguous'
    assert row['trust'] == 'source_bound_self_report'
    assert not row['verified_throughput'] and not row['bottleneck_verified']
    assert read(log(model_memory_format=layout, input_memory_format=layout, examples_per_second=999))['window'] is None


@pytest.mark.parametrize('layout', ['channels_first or channels_last', 'channels_first (contiguous NCHW) or NHWC'])
def test_ambiguous_descriptive_layout_stays_unknown(layout):
    assert read(log(model_memory_format=layout, input_memory_format=layout))['window'] is None


def test_confounded_metric_drop_does_not_block_future_layout_advice(monkeypatch):
    _pending.clear(); _blocked.clear()
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR', 'v86-confounding-regression')
    before, after = historical_pair()
    parent = NS(id='p', code=before, exec_time=94.24, metric=NS(value=.8105, maximize=True))
    child = NS(id='c', code=after, exec_time=68.67, is_buggy=False, metric=NS(value=.8095, maximize=True))
    frozen = copy.deepcopy(vars(child))
    remember(TASK, parent, child, {'vision_layout'}, 'bf16_channels_last')
    blocked, events = observe(TASK, before)
    assert blocked == [] and len(events) == 1
    event = events[0]
    assert event['recipe_confounded'] and not event['time_or_quality_comparison_eligible_source_screen']
    assert event['suppression_reasons'] == [] and not event['node_modified']
    assert vars(child) == frozen and not event['causal_gain_verified']


def test_failure_still_warns_even_if_recipe_changes(monkeypatch):
    _pending.clear(); _blocked.clear()
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR', 'v86-failure-regression')
    parent = NS(id='p', code=CODE, exec_time=1., metric=NS(value=.8, maximize=True))
    child = NS(id='c', code=CODE + '\nema=ModelEMA(model)', exec_time=2., is_buggy=True, metric=NS(value=None, maximize=True))
    remember(TASK, parent, child, {'vision_layout'}, 'bf16_channels_last')
    blocked, events = observe(TASK, CODE)
    assert blocked == ['bf16_channels_last']
    assert events[0]['recipe_confounded'] and 'attribution unknown' in events[0]['suppression_reasons'][0]


@pytest.mark.parametrize('phase', ['planner', 'coder'])
def test_compact_parent_advice_preserves_ema_selection(phase):
    text, audit = build(phase=phase)
    assert text and len(text) <= 750
    assert 'keep EMA/checkpoint selection unchanged' in text
    assert audit['policy_revision'] == 'v8.18'
    assert audit['added_provider_calls'] == 0
