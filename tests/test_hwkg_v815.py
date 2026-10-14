"""Model-path calibration scope and precise native log spelling regressions."""
import pytest
from hwkg.source_analysis import training_scope_uncertain
from hwkg.native_evidence import source_spec, native_feedback
from tests.test_hwkg_v814 import CODE, call
from tests.test_hwkg_v8 import GPU, log


@pytest.mark.parametrize('mutation',[
    'model.forward = replacement',
    'block.forward = types.MethodType(replacement, block)',
    'Model.forward = replacement',
    'model.__call__ = replacement',
    'del model.forward',
    "setattr(model, 'forward', replacement)",
    "delattr(model, 'forward')",
    "model.__setattr__('forward', replacement)",
    "model.__delattr__('forward')",
    'model.register_forward_hook(replacement)',
    'model.register_forward_pre_hook(replacement)',
    'model.register_full_backward_hook(replacement)',
    'model.register_full_backward_pre_hook(replacement)',
    'model.register_backward_hook(replacement)',
])
def test_modified_execution_path_does_not_borrow_ordinary_model_calibration(mutation):
    code=CODE+'\n'+mutation+'\n'
    assert training_scope_uncertain(code)
    assert source_spec(code)['training_scope']=='unknown'
    assert call(code) is None


@pytest.mark.parametrize('suffix',[
    '\n# model.forward = replacement\n',
    '\nprint("register_forward_hook")\n',
    '\ndef forward(x):\n    return x\n',
    '\nmodel.eval()\n',
])
def test_unmodified_ordinary_model_scope_remains_eligible(suffix):
    code=CODE+suffix
    assert not training_scope_uncertain(code)
    assert source_spec(code)['training_scope']=='full_model_source_screen'
    assert call(code) is not None


def test_precise_NCHW_log_spelling_is_source_bound_self_report_only():
    text=log(model_memory_format='channels_first(NCHW,contiguous)',
             input_memory_format='channels_first(NCHW,contiguous)')
    feedback=native_feedback(CODE,text,GPU,parent_id='p',execution_succeeded=True)
    assert feedback['trust']=='source_bound_self_report'
    assert feedback['window']['model_layout']=='contiguous'
    assert feedback['window']['input_layout']=='contiguous'
    assert not feedback['verified_throughput'] and not feedback['bottleneck_verified']


def test_ambiguous_layout_is_not_promoted_to_NHWC_or_verified_runtime():
    text=log(model_memory_format='non-contiguous',input_memory_format='non-contiguous')
    feedback=native_feedback(CODE,text,GPU,parent_id='p',execution_succeeded=True)
    assert feedback['window'] is None and feedback['trust']=='unknown'
    assert 'unknown native layouts' in feedback['reasons']
