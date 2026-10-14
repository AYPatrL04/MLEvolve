"""Scoped optimizer advice tests; fixtures are not task performance evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg.optimizer_policy import guidance, load_record, only_fused_flag_changed, FEATURE, BUNDLED_SHA256
from hwkg.native_evidence import source_spec
from hwkg.calibration import workload_key
from hwkg import trial_memory
from tests.test_hwkg_v8 import CODE as BASE, GPU, build
from tests.test_hwkg_v6 import GRAPH,IDS,FEATURES

CODE=BASE.replace('1e-3','0.0003').replace('1e-4','0.05').replace('logits,yb)','logits,yb,label_smoothing=0.1)').replace('model=model.cuda()', 'model=model.cuda().to(memory_format=torch.channels_last)')
CODE=CODE.replace('import torch\n','import torch\ntorch.backends.cudnn.benchmark=True\ntorch.backends.cuda.matmul.allow_tf32=True\ntorch.backends.cudnn.allow_tf32=True\n')
RECORD=load_record()
RUNTIME={**RECORD['host'],'bf16_supported':True,'calibration_flags':RECORD['runtime_flags'],'free_mib_snapshot':12000}
EDGES={e['to']:e for e in GRAPH['edges'] if e['from']==IDS['5080']}
TASK='CNN classification; train.csv (40000 rows); six epochs'


def call(code=CODE, **kwargs):
    return guidance(kwargs.pop('stage','improve'),kwargs.pop('task',TASK),code,GPU,
                    kwargs.pop('edges',EDGES),kwargs.pop('runtime',RUNTIME),
                    parent_execution_succeeded=kwargs.pop('parent_execution_succeeded',True),**kwargs)


@pytest.mark.parametrize('phase',['planner','coder','plan_and_code'])
def test_measured_scope_proposal_is_compact_conditional_and_single_flag(phase):
    text,audit=call(phase=phase)
    assert len(text)<=750 and 'fused=True' in text and 'Preserve AMP, batch, LR/WD' in text
    assert audit['policy_revision']=='v8.16' and audit['advice_ids']==['optimizer_fused_adamw']
    assert audit['calibration_file_sha256']==BUNDLED_SHA256 and audit['projected_updates']==3750
    assert audit['calibration_trial']['repeat_spread_net_seconds']>0
    assert audit['earlier_short_repeat_spread_conflict_retained']
    assert not audit['end_to_end_gain_verified'] and not audit['measurements_verified']
    assert audit['added_provider_calls']==0 and not audit['repeat_spread_is_CI']


@pytest.mark.parametrize('old,new',[
    ('bfloat16','float16'),('resnet50','resnet18'),('BATCH=64','BATCH=128'),('SIZE=128','SIZE=224'),
    ('lr=0.0003','lr=.001'),('weight_decay=0.05','weight_decay=.01'),('label_smoothing=0.1','label_smoothing=0.'),
    ('EPOCHS=6','EPOCHS=3'),('EPOCHS=6','EPOCHS=7'),('ResNet50_Weights.IMAGENET1K_V2','None'),
    ('model.fc.in_features,100','model.fc.in_features,200'),('torch.channels_last','torch.contiguous_format'),
    ('cudnn.benchmark=True','cudnn.benchmark=False'),('matmul.allow_tf32=True','matmul.allow_tf32=False'),
    ('weight_decay=0.05','weight_decay=0.05,fused=True'),('model.parameters()','[model.fc.weight]'),
])
def test_unmatched_recipe_dtype_budget_or_already_fused_not_offered(old,new):
    assert call(CODE.replace(old,new)) is None


@pytest.mark.parametrize('field,value',[('torch','2.8.0'),('cuda','12.0'),('bf16_supported',False),('free_mib_snapshot',100),('free_mib_snapshot',0)])
def test_runtime_memory_identity_required(field,value):
    assert call(runtime={**RUNTIME,field:value}) is None


def test_graph_edge_is_a_scoped_host_entry_and_is_required():
    assert EDGES[FEATURE]['properties']['verified']
    e=copy.deepcopy(EDGES);e[FEATURE]['properties']['verified']=False
    assert call(edges=e) is None
    e=copy.deepcopy(EDGES);e.pop(FEATURE);assert call(edges=e) is None
    assert call(task='train.csv (50000 rows)') is None
    assert call(task='train.csv (40000 rows) train.csv (1 rows)') is None


def test_actual_provider_row_identity_can_restore_cleaned_task_metadata():
    assert call(task='cleaned CNN task',provider_prompt={'assistant':TASK}) is not None
    assert call(task='cleaned CNN task',provider_prompt={'assistant':'train.csv (1000 rows)'}) is None


@pytest.mark.parametrize('kwargs',[{'stage':'debug'},{'phase':'merge'},{'parent_execution_succeeded':False},
    {'parent_execution_succeeded':None},{'blocked_profiles':['optimizer_fused_adamw']},{'max_chars':100}])
def test_stage_parent_failure_suppression_and_budget(kwargs):
    assert call(**kwargs) is None


@pytest.mark.parametrize('reserve',['10','nan','inf','-1','bad'])
def test_nonprofitable_or_invalid_declared_cost_not_offered(monkeypatch,reserve):
    monkeypatch.setenv('MLEVOLVE_HWKG_COST_RESERVE_SECONDS',reserve)
    assert call() is None


def test_unrecognized_transform_or_mixed_scaler_domain_not_offered():
    assert call(CODE+'\nmodel=torch.compile(model)') is None
    active=CODE.replace('with torch.autocast', 'scaler=torch.amp.GradScaler("cuda")\nwith torch.autocast').replace('loss.backward()\noptimizer.step()', 'scaler.scale(loss).backward()\nscaler.step(optimizer)\nscaler.update()')
    assert call(active) is None
    assert call('') is None


def test_guidance_integrates_into_component_only_without_using_printed_throughput():
    text,audit=build(code=CODE,runtime=RUNTIME,failure_text='',phase='coder')
    assert text and audit['advice_ids']==['optimizer_fused_adamw']
    assert audit['native_parent_evidence']['window'] is None
    assert audit['calibration_trial']['change']=='optimizer_implementation'


def test_exact_flag_only_syntax_screen_and_no_extra_changes():
    after=CODE.replace('weight_decay=0.05)','weight_decay=0.05,fused=True)')
    assert only_fused_flag_changed(CODE,after)
    assert not only_fused_flag_changed(CODE,after.replace('EPOCHS=6','EPOCHS=5'))
    assert not only_fused_flag_changed(CODE,after+'\nprint("extra")')
    assert not only_fused_flag_changed(CODE,after.replace('fused=True','fused=ENABLED'))
    assert not only_fused_flag_changed('',after)


def test_negative_runtime_observation_warns_on_exact_flag_change_without_node_mutation(monkeypatch):
    monkeypatch.setattr(trial_memory,'_pending',{});monkeypatch.setattr(trial_memory,'_blocked',{})
    after=CODE.replace('weight_decay=0.05)','weight_decay=0.05,fused=True)')
    parent=NS(id='p',code=CODE,exec_time=10.,metric=NS(value=.8,maximize=True))
    child=NS(id='c',code=after,exec_time=12.,metric=NS(value=.7,maximize=True),is_buggy=False)
    snapshot=copy.deepcopy(child.__dict__)
    trial_memory.remember(TASK,parent,child,{'optimizer_fused_adamw'},'optimizer_fused_adamw')
    blocked,events=trial_memory.observe(TASK,CODE)
    assert child.__dict__==snapshot and not events[0]['node_modified']
    assert blocked==['optimizer_fused_adamw'] and len(events[0]['suppression_reasons'])==2
    assert events[0]['time_or_quality_comparison_eligible_source_screen']
    assert not events[0]['causal_gain_verified']


def test_other_recipe_change_not_blame_fused_profile(monkeypatch):
    monkeypatch.setattr(trial_memory,'_pending',{});monkeypatch.setattr(trial_memory,'_blocked',{})
    after=CODE.replace('weight_decay=0.05)','weight_decay=0.05,fused=True)').replace('EPOCHS=6','EPOCHS=5')
    parent=NS(id='p',code=CODE,exec_time=10.,metric=NS(value=.8,maximize=True))
    child=NS(id='c',code=after,exec_time=12.,metric=NS(value=.7,maximize=True),is_buggy=False)
    trial_memory.remember(TASK,parent,child,{'optimizer_fused_adamw'},'optimizer_fused_adamw')
    blocked,events=trial_memory.observe(TASK,CODE)
    assert blocked==[] and events[0]['recipe_confounded']


def test_scoped_dispatch_tracks_new_profile_without_request_or_native_node_changes(monkeypatch):
    from hwkg import routing
    from hwkg.routing import hardware_context,with_hardware_prompt,_context
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH','fixture-graph-path')
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v8')
    text,_=call();events=[];calls=[];cfg=NS(agent=NS(code=NS(model='fixture-model')))
    monkeypatch.setattr(routing,'hardware_guidance',lambda *a,**k:text)
    monkeypatch.setattr(routing,'_audit',lambda row:events.append(row))
    monkeypatch.setattr(trial_memory,'_pending',{});monkeypatch.setattr(trial_memory,'_blocked',{})
    @with_hardware_prompt
    def provider(prompt,cfg,**kwargs):
        calls.append((prompt,cfg,kwargs));return 'native fixture answer'
    class Agent:
        task_desc=TASK
        @hardware_context('improve')
        def run(self,parent_node):
            provider('Code prompt',cfg,max_tokens=16384)
            self.target=_context.get().target_profile
            return NS(id='own-child',code=CODE.replace('weight_decay=0.05)','weight_decay=0.05,fused=True)'))
    parent=NS(id='p',code=CODE,is_buggy=False,exc_type=None,_term_out=[],metric=NS(value=.8,maximize=True))
    frozen=copy.deepcopy(parent.__dict__);agent=Agent();node=agent.run(parent)
    assert parent.__dict__==frozen and agent.target=='optimizer_fused_adamw'
    assert len(calls)==1 and calls[0][1] is cfg and calls[0][2]=={'max_tokens':16384}
    returned=next(x for x in events if x['event']=='candidate_return')
    assert returned['optimizer_implementation_observation']=={'fused_source_option':True,'runtime_verified':False,'gain_verified':False}
    assert only_fused_flag_changed(parent.code,node.code)
