import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from hwkg.native_evidence import native_feedback,source_spec,recipe_changes
from hwkg.calibration import workload_key,select_record,choose_trial,SCHEMA,calibration_path
from hwkg.feedback_policy import guidance_v8
from hwkg.routing import hardware_context,hardware_phase,with_hardware_prompt,_append_feedback_advice
from tests.test_hwkg_v6 import GRAPH,FEATURES,DEVICES,IDS,RUNTIME,CFG

GPU=DEVICES['5080']
CODE="""import torch
from torchvision.models import resnet50
BATCH=64
SIZE=128
EPOCHS=6
model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.fc=torch.nn.Linear(model.fc.in_features,100)
model=model.cuda()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
with torch.autocast('cuda',dtype=torch.bfloat16):
    logits=model(xb)
    loss=torch.nn.functional.cross_entropy(logits,yb)
loss.backward()
optimizer.step()
"""
TASK='CNN image classification; train.csv (40000 rows); six epochs'

def log(**changes):
    row={'optimizer_steps':20,'completed_examples':1280,'window_start_step':10,'window_end_step':30,
         'skipped_updates':0,'wall_seconds':.5,'examples_per_second':2560,
         'actual_autocast_dtype':'bfloat16','model_memory_format':'torch.contiguous_format',
         'input_memory_format':'torch.contiguous_format','peak_allocated_bytes':2*1024**3}
    row.update(changes);return 'THROUGHPUT_JSON '+json.dumps(row)

def read(text=None,code=CODE):
    return native_feedback(code,log() if text is None else text,GPU,parent_id='p',execution_succeeded=True)

def build(code=CODE,phase='coder',stage='improve',**kwargs):
    node=copy.deepcopy(next(n for n in GRAPH['nodes'] if n['id']==IDS['5080']))
    edges={e['to']:copy.deepcopy(e) for e in GRAPH['edges'] if e['from']==IDS['5080']}
    runtime=kwargs.pop('runtime',RUNTIME)
    return guidance_v8(stage,TASK,code,GPU,node,FEATURES,edges,runtime,phase=phase,
        parent_id='p',parent_execution_succeeded=kwargs.pop('parent_execution_succeeded',True),
        failure_text=kwargs.pop('failure_text',log()),**kwargs)

def calibration(tmp_path):
    row={'workload':workload_key(GPU,RUNTIME,source_spec(CODE)),'quality_verified':False,'profiles':{}}
    for name,t,setup in [('fp32_contiguous',.02,1.),('bf16_contiguous',.01,1.),('bf16_channels_last',.006,1.1)]:
        row['profiles'][name]={'step_seconds':t,'setup_seconds':setup,'normal_updates':25,'repeats':3,'finite_updates':True,
                              'finite_check_scope':'all_losses_initial_grad_final_parameters'}
    path=tmp_path/'calibration.json';path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}));return path,row

def test_native_math_is_source_bound_but_never_verified_gain():
    row=read();assert row['trust']=='source_bound_self_report' and row['window']['examples_per_second']==2560
    assert not row['verified_throughput'] and not row['bottleneck_verified']
    assert row['source_spec']['forwards_per_update']==1

@pytest.mark.parametrize('change',[{'wall_seconds':0},{'wall_seconds':float('nan')},{'optimizer_steps':True},
    {'optimizer_steps':19},{'completed_examples':64},{'examples_per_second':99999},{'window_end_step':28},
    {'code_sha256':'wrong'},{'candidate_id':'other'},{'gpu':{'name':'NVIDIA A10','capability':[8,6]}},
    {'actual_autocast_dtype':'invented'},{'input_memory_format':'invented'},
    {'model_memory_format':[]},{'completed_examples':10**500}])
def test_bad_native_window_is_unknown(change):
    assert read(log(**change))['window'] is None

@pytest.mark.parametrize('text',['','THROUGHPUT_JSON []','THROUGHPUT_JSON nope',log()+'\n'+log()])
def test_missing_or_ambiguous_evidence(text):
    assert read(text)['window'] is None

def test_work_scope_and_parent_failure_do_not_become_proof():
    assert read(log(training_scope='head_only'))['reasons']
    assert read(log(skipped_updates=1,window_end_step=31))['reasons']
    assert source_spec(CODE+'\noptimizer.first_step();optimizer.second_step()')['forwards_per_update']==2
    assert source_spec(CODE+'\nfor p in model.parameters():p.requires_grad_(False)')['training_scope']=='unknown'
    for status in [False,None]:assert native_feedback(CODE,log(),GPU,execution_succeeded=status)['window'] is None

@pytest.mark.parametrize('field',['gpu_name','torch','cuda','batch','input_shape','optimizer','forwards_per_update',
                                 'weights','classifier_classes','loss','optimizer_options','runtime_flags'])
def test_calibration_exact_scope(tmp_path,field):
    path,row=calibration(tmp_path);row['workload'][field]='wrong';path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}))
    assert select_record(path,GPU,RUNTIME,source_spec(CODE))[0] is None

@pytest.mark.parametrize('bad',[{'repeats':1},{'normal_updates':10},{'step_seconds':float('inf')},
                              {'setup_seconds':-1},{'finite_updates':False}])
def test_calibration_rejects_invalid_trials(tmp_path,bad):
    path,row=calibration(tmp_path);row['profiles']['bf16_contiguous'].update(bad)
    path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}))
    assert select_record(path,GPU,RUNTIME,source_spec(CODE))[0] is None

def test_calibration_roi_and_no_symlink_reads(tmp_path):
    path,row=calibration(tmp_path);link=tmp_path/'link.json';link.symlink_to(path)
    assert select_record(link,GPU,RUNTIME,source_spec(CODE))[0] is None
    assert select_record(path,GPU,RUNTIME,source_spec(CODE))[0]
    assert choose_trial(row,'bf16_contiguous',1000)['profile']=='bf16_channels_last'
    assert choose_trial(row,'bf16_contiguous',1000,5) is None
    row['profiles']['bf16_channels_last']['setup_seconds']=100
    assert choose_trial(row,'bf16_contiguous',1000) is None
    assert choose_trial(row,'bf16_channels_last',1000) is None

@pytest.mark.parametrize('change',[('weight_decay=1e-4','weight_decay=1e-2'),
    ('weight_decay=1e-4','weight_decay=1e-4,fused=True'),
    ('model.parameters()','[model.fc.weight]'),
    ('model.fc.in_features,100','model.fc.in_features,1000'),
    ('ResNet50_Weights.IMAGENET1K_V2','None'),
    ('logits,yb','logits,yb,label_smoothing=0.1'),
    ('import torch','import torch\ntorch.backends.cudnn.benchmark=True')])
def test_different_recipes_do_not_borrow_microcalibration(tmp_path,change):
    path,_=calibration(tmp_path)
    assert select_record(path,GPU,RUNTIME,source_spec(CODE.replace(*change)))[0] is None

def test_malformed_record_collection_is_ignored(tmp_path):
    path,_=calibration(tmp_path);path.write_text(json.dumps({'schema':SCHEMA,'records':None}))
    assert select_record(path,GPU,RUNTIME,source_spec(CODE))[0] is None

def test_bundled_calibration_is_scoped_and_optional(monkeypatch):
    monkeypatch.delenv('MLEVOLVE_HWKG_CALIBRATION_PATH',raising=False)
    path=calibration_path(GPU);doc=json.loads(path.read_text());key=doc['records'][0]['workload']
    runtime={'torch':key['torch'],'cuda':key['cuda'],'calibration_flags':key['runtime_flags']}
    assert select_record(path,GPU,runtime,source_spec(CODE))[0]
    assert select_record(path,GPU,{**runtime,'cuda':'other'},source_spec(CODE))[0] is None
    assert calibration_path(DEVICES['a10']) is None
    monkeypatch.setenv('MLEVOLVE_HWKG_CALIBRATION_PATH','')
    assert calibration_path(GPU)=='' and select_record('',GPU,runtime,source_spec(CODE))[0] is None

def test_planner_and_coder_have_different_roles():
    planner,pa=build(phase='planner');coder,ca=build()
    assert planner and coder and planner!=coder
    assert 'Estimate setup' in planner and 'model.to(memory_format=torch.channels_last)' in coder
    assert 'counter math checked' in coder and not ca['measurements_verified']
    assert 'never a .memory_format attribute' in coder and len(coder)<=1100
    assert ca['advice_ids']==['vision_layout'] and 'original response format' in coder
    for text,audit in [(planner,pa),(coder,ca)]:
        assert 'No extra training/benchmark/BN-stat passes' in text
        assert len(text)<=750
        assert audit['policy_revision']=='v8.16'

def test_measured_profile_roi_and_native_scope_gate(tmp_path,monkeypatch):
    path,row=calibration(tmp_path);monkeypatch.setenv('MLEVOLVE_HWKG_CALIBRATION_PATH',str(path))
    text,audit=build();assert text and audit['calibration_trial']['estimated_net_seconds']>0
    assert 'Quality/full-pipeline gain unverified' in text
    assert build(failure_text=log(training_scope='head_only'))[1]['calibration_trial'] is None
    row['profiles']['bf16_channels_last']['step_seconds']=.011;path.write_text(json.dumps({'schema':SCHEMA,'records':[row]}))
    text,audit=build();assert text is None and 'no profitable' in audit['skip_reason']

def test_actual_reported_dtype_layout_bound_calibration(tmp_path,monkeypatch):
    path,_=calibration(tmp_path);monkeypatch.setenv('MLEVOLVE_HWKG_CALIBRATION_PATH',str(path))
    _,audit=build(failure_text=log(actual_autocast_dtype='float16'))
    assert audit['current_profile']=='fp16_contiguous' and audit['calibration_trial'] is None
    _,audit=build(failure_text=log(model_memory_format='channels_last'))
    assert audit['current_profile'] is None and audit['calibration_trial'] is None
    _,audit=build(failure_text=log(model_memory_format='channels_last',input_memory_format='channels_last'))
    assert audit['current_profile']=='bf16_channels_last' and audit['calibration_trial'] is None

def test_adopted_layout_merge_and_general_debug_are_noops():
    assert build(code=CODE+'\nmodel.to(memory_format=torch.channels_last)',
                 failure_text=log(model_memory_format='channels_last',input_memory_format='channels_last'))[0] is None
    assert build(phase='merge')[0] is None
    assert build(stage='debug',failure_text='AttributeError: SAM has no attribute adaptive')[0] is None
    assert build(parent_execution_succeeded=False)[0] is None

def test_native_report_can_contradict_unused_source_layout_flag():
    text,audit=build(code=CODE+'\nif False:model.to(memory_format=torch.channels_last)')
    assert text and audit['advice_ids']==['vision_layout']
    assert not audit['measurements_verified'] and 'self-report' in text

def test_v8_ignores_untrusted_legacy_adapter_records():
    text,audit=build(failure_text='MLEVOLVE_RUNTIME_JSON []\n'+log())
    assert text and audit['advice_ids']==['vision_layout']
    assert audit['native_parent_evidence']['window']

def test_precision_requires_real_bf16_support():
    code=CODE.replace("with torch.autocast('cuda',dtype=torch.bfloat16):",'if True:')
    assert build(code=code,runtime={**RUNTIME,'bf16_supported':False},
                 failure_text=log(actual_autocast_dtype='float32'))[0] is None

@pytest.mark.parametrize('phase',['coder','plan_and_code'])
def test_precision_path_also_guards_layout_reporting(phase):
    code=CODE.replace("with torch.autocast('cuda',dtype=torch.bfloat16):",'if True:')
    text,audit=build(code=code,phase=phase,failure_text=log(actual_autocast_dtype='float32'))
    assert text and audit['advice_ids']==['precision']
    assert 'is_contiguous(memory_format=torch.channels_last)' in text
    assert 'never a .memory_format attribute' in text
    assert len(text)<=1100 and audit['policy_revision']=='v8.16'
    assert audit['added_provider_calls']==0 and not audit['measurements_verified']
    assert 'TRAIN forward/loss only' in text and 'eval/test precision' in text
    assert 'RNG/sample order' in text and 'shared eval autocast helper unchanged' in text

def test_layout_report_expression_works_on_parameters_and_inputs():
    import torch
    for tensor in [torch.zeros(2,3,4,4),torch.nn.Parameter(torch.zeros(2,3,4,4))]:
        assert not tensor.is_contiguous(memory_format=torch.channels_last)
        converted=tensor.to(memory_format=torch.channels_last)
        assert converted.is_contiguous(memory_format=torch.channels_last)
        assert not hasattr(tensor,'memory_format')

def test_output_format_and_chat_prefill_preserved():
    prompt='Task\n# Response format\nOriginal exact fences.'
    text=_append_feedback_advice(prompt,'trial')
    assert text.endswith('# Response format\nOriginal exact fences.')
    chat={'system':'system','user':prompt,'assistant':'prefill'};out=_append_feedback_advice(chat,'trial')
    assert chat['user']==prompt and out['system']=='system' and out['assistant']=='prefill'
    assert _append_feedback_advice('original','trial').endswith('original')

@pytest.fixture
def enabled(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    for key,val in {'GRAPH_PATH':str(root/'hwkg/graph.json'),'GRAPH_SHA256':'3842eff0a7742c42b3777a01a6f58cfbec42b37c442d9bb9f5b080321ff7ed62',
                    'HARDWARE_ID':IDS['5080'].removeprefix('hw:'),'AUDIT_DIR':str(tmp_path)}.items():monkeypatch.setenv('MLEVOLVE_HWKG_'+key,val)
    monkeypatch.delenv('MLEVOLVE_HWKG_EVIDENCE_POLICY',raising=False)
    monkeypatch.delenv('MLEVOLVE_HWKG_CALIBRATION_PATH',raising=False)
    with patch('hwkg.injection._visible_cuda_device',return_value=GPU),patch('hwkg.host_policy.runtime_facts',return_value=RUNTIME):yield tmp_path

def test_retry_decision_cache_keeps_advice_provider_calls_and_phase(enabled):
    sent=[]
    @with_hardware_prompt
    def send(p,cfg):sent.append(p);return 'unchanged response'
    @hardware_phase('planner')
    def plan():send('plan',CFG);send('retry',CFG);return 'same recipe'
    @hardware_phase('coder')
    def code():send('code\n# Response format\nOriginal format.',CFG)
    @hardware_context('improve')
    def run(agent,parent_node):plan();code();return NS(id='new',code=CODE+'\nmodel.to(memory_format=torch.channels_last)')
    run(NS(task_desc=TASK),NS(id='p',code=CODE,_term_out=[log()],exc_type=None,is_buggy=False))
    assert len(sent)==3 and 'HWKG v8' in sent[0] and 'HWKG v8' in sent[1]
    assert sent[1].endswith('retry')
    assert 'model.to(memory_format=' in sent[2] and sent[2].endswith('Original format.')
    rows=[json.loads(x) for x in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert sum(x['event']=='provider_dispatch' for x in rows)==3
    assert sum(x['event']=='context_cache_hit' for x in rows)==1
    ret=next(x for x in rows if x['event']=='candidate_return')
    assert ret['intervention_observations']['vision_layout']['source_configured']
    assert not ret['intervention_observations']['vision_layout']['runtime_verified']
    assert ret['recipe_scope']['recipe_syntax_changed'] is False
    send('outside',CFG);assert sent[-1]=='outside'

def test_reused_prompt_delivers_one_advisory_block_and_preserves_format():
    original='Task\n# Response format\nOriginal exact fences.'
    first=_append_feedback_advice(original,'trial')
    assert _append_feedback_advice(first,'trial')==first
    doubled=first.replace('\n# Hardware knowledge (advisory)\ntrial\n\n',
        '\n# Hardware knowledge (advisory)\ntrial\n\n'*2)
    assert _append_feedback_advice(doubled,'trial')==first
    chat={'system':'original system','user':first,'assistant':'original prefill'}
    assert _append_feedback_advice(chat,'trial')==chat

def test_failed_provider_retry_receives_same_cached_hardware_decision(enabled):
    sent=[]
    @with_hardware_prompt
    def send(prompt,cfg):
        sent.append(prompt)
        if len(sent)==1:raise ConnectionError('temporary transport failure')
        return 'unchanged response'
    @hardware_phase('coder')
    def coding():
        prompt='Task\n# Response format\nOriginal format.'
        with pytest.raises(ConnectionError):send(prompt,CFG)
        return send(prompt,CFG)
    @hardware_context('improve')
    def run(agent,parent_node):return coding()
    assert run(NS(task_desc=TASK),NS(id='p',code=CODE,_term_out=[log()],exc_type=None,is_buggy=False))=='unchanged response'
    assert len(sent)==2 and sent[0]==sent[1] and 'HWKG v8' in sent[1]
    rows=[json.loads(x) for x in (enabled/'dispatch-audit.jsonl').read_text().splitlines()]
    assert sum(r['event']=='provider_failure' for r in rows)==1
    assert sum(r['event']=='provider_dispatch' for r in rows)==2
    assert sum(r['event']=='context_cache_hit' for r in rows)==1

def test_recipe_drift_is_only_audited():
    assert recipe_changes(CODE,CODE.replace('lr=1e-3','lr=1e-2'))['recipe_syntax_changed']
    assert not recipe_changes(CODE,CODE+'\nmodel.to(memory_format=torch.channels_last)')['recipe_syntax_changed']
    assert not recipe_changes(CODE,CODE)['semantic_preservation_certified']

def test_actual_native_study_logs_now_consumable():
    root=Path(__file__).resolve().parents[2]/'MLEvolve/records/hwkg-v73-native-5080-cifar100-local-s42-15nodes-20261012'
    for arm,node in [('baseline','5141b3438ef644ebabc2d6f47c5bc459'),('treatment','e898ba5ef1134b22b3fa4cd84beaf5bb')]:
        p=root/arm/'candidates'/node
        if not p.exists():pytest.skip('historical workstation evidence not present')
        result=native_feedback((p/'candidate.py').read_text(),(p/'execution.log').read_text(),GPU,parent_id=node,execution_succeeded=True)
        assert result['trust']=='source_bound_self_report' and result['window']
        assert not result['verified_throughput']
        if arm=='treatment':assert result['source_spec']['forwards_per_update']==2
