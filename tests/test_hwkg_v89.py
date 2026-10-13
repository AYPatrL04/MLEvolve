"""Regression from actual native source; replay only, no candidate modifications."""
import hashlib
from pathlib import Path
import pytest
from hwkg.native_evidence import recipe_changes


def native_source():
    path = Path(__file__).resolve().parents[2] / (
        'MLEvolve/records/hwkg-v88-native-5080-cifar100-local-s42-light4-20261013T151133Z/'
        'treatment/candidates/fd37647268634519a8ae5c5d8404eea4/candidate.py')
    if not path.exists(): pytest.skip('saved native public source unavailable')
    source = path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == '5947ceec87dfb1e0e7928d053bb5d0c10ef706ae713e89837d75e3ad67a8c197'
    return source


@pytest.mark.parametrize('old,new', [
    ('BACKBONE_LR, HEAD_LR = 2.5e-4, 2.5e-3', 'BACKBONE_LR, HEAD_LR = 5e-4, 2.5e-3'),
    ('BATCH, NW = 128, 4', 'BATCH, NW = 256, 4'),
    ('smooth = -logp.mean(dim=1)', 'smooth = -logp.sum(dim=1)'),
])
def test_actual_recipe_mutations_are_not_misclassified_as_hardware_only(old, new):
    source = native_source()
    assert old in source
    report = recipe_changes(source, source.replace(old, new))
    assert report['recipe_syntax_changed'] is True
    assert report['semantic_preservation_certified'] is False


def test_referenced_group_settings_follow_aliases_and_annotated_assignments():
    source = 'BASE: float = .001\nRATE=BASE\nGROUPS=[{"params": parameters, "lr":RATE}]\nopt=torch.optim.AdamW(GROUPS)\n'
    assert recipe_changes(source, source.replace('.001', '.002'))['recipe_syntax_changed']


def test_custom_module_constructor_setting_and_formula_are_screened():
    source = 'EPS=.1\nclass SmoothCE(nn.Module):\n    def forward(self, x, y): return x.mean()\ncrit=SmoothCE(EPS)\n'
    assert recipe_changes(source, source.replace('EPS=.1', 'EPS=.2'))['recipe_syntax_changed']
    assert recipe_changes(source, source.replace('x.mean()', 'x.sum()'))['recipe_syntax_changed']


def test_layout_and_worker_changes_do_not_change_quality_recipe_in_real_source():
    source = native_source()
    workers = source.replace('BATCH, NW = 128, 4', 'BATCH, NW = 128, 2')
    layout = source.replace('model = model.cuda().to(memory_format=torch.channels_last)', 'model = model.cuda()')
    assert workers != source and layout != source
    assert not recipe_changes(source, workers)['recipe_syntax_changed']
    assert not recipe_changes(source, layout)['recipe_syntax_changed']


def test_cyclic_settings_remain_finite_and_are_not_semantically_certified():
    source = 'LR=RATE\nRATE=LR\nopt=torch.optim.AdamW(params,lr=LR)\n'
    report = recipe_changes(source, source)
    assert not report['recipe_syntax_changed']
    assert report['semantic_preservation_certified'] is False
