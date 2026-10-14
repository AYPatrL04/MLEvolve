"""Catalogue-wide dispatch fixtures and host-bound numerical evidence checks."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from hwkg import injection
from hwkg.calibration import calibration_path, select_record, workload_key
from hwkg.device_support import architecture, options, AMP_TRIAL, FUSED_TRIAL
from hwkg.exploration_policy import guidance
from hwkg.feedback_policy import guidance_v8
from hwkg.optimizer_policy import guidance as measured_optimizer, load_record
from tests.test_hwkg_v814 import CODE, RUNTIME as MEASURED_RUNTIME, GPU as LOCAL_GPU
from hwkg.native_evidence import source_spec

ROOT=Path(__file__).resolve().parents[1]
GRAPH=json.loads((ROOT/'hwkg/graph.json').read_text())
SHA=hashlib.sha256((ROOT/'hwkg/graph.json').read_bytes()).hexdigest()
HOSTS=[n for n in GRAPH['nodes'] if n['id'].startswith('hw:')]
FEATURES={n['id']:n for n in GRAPH['nodes'] if n['id'].startswith('feat:')}


def setup_host(node):
    props=node['properties'];cap=props['compute_capabilities'][0]
    kw=dict(name=props['name'],total_memory=props.get('vram_per_gpu_MB',props['vram_MB'])*1024**2)
    rocm=cap.startswith('gfx')
    if rocm:kw.update(major=0,minor=0,gcnArchName=cap+':sramecc-:xnack-')
    else:kw.update(major=int(cap.split('.')[0]),minor=int(cap.split('.')[1]))
    gpu=NS(**kw)
    edges={e['to']:copy.deepcopy(e) for e in GRAPH['edges'] if e['from']==node['id']}
    runtime=dict(torch='test-rocm' if rocm else 'test-cuda',cuda=None if rocm else '13.0',
                 hip='7.2' if rocm else None,backend='rocm' if rocm else 'cuda',
                 bf16_supported=rocm or gpu.major>=8,amp_api=True,grad_scaler_api=True,
                 adamw_fused_api=True,adamw_fused_kernel=True,
                 free_mib_snapshot=kw['total_memory']//1024**2,
                 calibration_flags={},sdpa_api=True,transformer_engine_package=False,torchao_package=False)
    return gpu,edges,runtime


@pytest.mark.parametrize('node',HOSTS,ids=lambda n:n['id'])
def test_each_catalogue_profile_matches_its_member_and_receives_draft_advice(node):
    gpu,edges,runtime=setup_host(node)
    assert injection._matches_visible_device(node,gpu)
    text,audit=guidance('draft','',gpu,edges,runtime,phase='plan_and_code',parent_execution_succeeded=None)
    assert architecture(gpu) in text and len(text)<=340
    assert audit['hardware_fact_values']['gpu_name']==gpu.name
    assert audit['hardware_fact_values']['total_vram_mib']==gpu.total_memory//1024**2
    assert audit['calibration_trial'] is None and audit['added_provider_calls']==0
    if runtime['bf16_supported']:assert 'BF16' in text and 'fused=True' in text
    else:assert 'BF16' not in text and 'FP16' in text and 'GradScaler' in text
    if node['id'].startswith('hw:amd.'):
        assert 'gfx' in text and 'ROCm' in text and 'sm_' not in text
        assert 'CUDA13' not in text and 'TF32' not in text


@pytest.mark.parametrize('node',HOSTS,ids=lambda n:n['id'])
def test_actual_component_delivery_path_covers_catalogue(node,monkeypatch):
    gpu,edges,runtime=setup_host(node)
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH',str(ROOT/'hwkg/graph.json'))
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_SHA256',SHA)
    monkeypatch.setenv('MLEVOLVE_HWKG_HARDWARE_ID',node['id'].removeprefix('hw:'))
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v8')
    monkeypatch.delenv('MLEVOLVE_HWKG_AUDIT_DIR',raising=False)
    monkeypatch.setattr(injection,'_visible_cuda_device',lambda:gpu)
    monkeypatch.setattr('hwkg.host_policy.runtime_facts',lambda:runtime)
    monkeypatch.setattr('hwkg.calibration.calibration_flags',lambda:{})
    text=injection.hardware_guidance('draft','CNN classification; train.csv (4000 rows)')
    assert text and architecture(gpu) in text and 'v8.19' in text


@pytest.mark.parametrize('node',HOSTS,ids=lambda n:n['id'])
def test_parent_trials_use_host_target_and_never_borrow_5080_timing(node):
    gpu,edges,runtime=setup_host(node)
    code=CODE if runtime['bf16_supported'] else CODE.replace('bfloat16','float16')
    result=guidance('improve',code,gpu,edges,runtime,phase='coder',parent_execution_succeeded=True)
    if runtime['bf16_supported']:
        assert result and 'fused=True' in result[0] and architecture(gpu) in result[0]
    else:
        assert result is None
    record,basis=select_record(calibration_path(gpu),gpu,runtime,source_spec(code))
    assert record is None
    assert measured_optimizer('improve','train.csv (40000 rows)',code,gpu,edges,runtime,
                              parent_execution_succeeded=True) is None


@pytest.mark.parametrize('field',['amp_api','adamw_fused_api','adamw_fused_kernel','bf16_supported'])
def test_runtime_gates_do_not_inherit_5080_capabilities(field):
    node=next(n for n in HOSTS if n['id'].endswith('geforce_rtx_4090.spec'))
    gpu,edges,runtime=setup_host(node);runtime[field]=False
    text,audit=guidance('draft','',gpu,edges,runtime,phase='coder',parent_execution_succeeded=None)
    if field=='amp_api':assert 'BF16' not in text and 'FP16' not in text and 'FP32' in text
    assert 'fused=True' not in text
    assert audit['calibration_trial'] is None


def test_T4_never_borrows_emulated_BF16_and_missing_edges_suppress_trials():
    node=next(n for n in HOSTS if n['id'].endswith('tesla_t4.spec'))
    gpu,edges,runtime=setup_host(node);runtime['bf16_supported']=True
    assert options(gpu,edges,runtime)['bf16'] is False
    text,_=guidance('draft','',gpu,edges,runtime,phase='coder',parent_execution_succeeded=None)
    assert 'BF16' not in text and 'GradScaler' in text
    edges.pop(AMP_TRIAL);edges.pop('feat:amp');edges.pop(FUSED_TRIAL)
    assert not options(gpu,edges,runtime)['amp']


def test_rocm_name_architecture_memory_and_vendor_are_all_required():
    node=next(n for n in HOSTS if n['id'].endswith('radeon_rx_7900_xtx.spec'))
    gpu,edges,runtime=setup_host(node)
    assert not injection._matches_visible_device(node,NS(**{**vars(gpu),'gcnArchName':'gfx1201'}))
    assert not injection._matches_visible_device(node,NS(**{**vars(gpu),'total_memory':8*1024**3}))
    assert not injection._matches_visible_device(node,NS(**{**vars(gpu),'name':'AMD Radeon RX 7900 XT'}))
    assert not injection._matches_visible_device(node,NS(name=gpu.name,major=12,minor=0,total_memory=gpu.total_memory))


def test_owned_A10_optimizer_calibration_matches_A10_only(tmp_path,monkeypatch):
    record=load_record();node=next(n for n in HOSTS if n['id'].endswith('a10.spec'))
    gpu,edges,runtime=setup_host(node)
    record['host'].update(gpu_name=gpu.name,capability=[gpu.major,gpu.minor])
    runtime={**MEASURED_RUNTIME,**{'bf16_supported':True,'adamw_fused_api':True,'adamw_fused_kernel':True}}
    file=tmp_path/'a10-optimizer.json';file.write_text(json.dumps(record))
    monkeypatch.setenv('MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH',str(file))
    result=measured_optimizer('improve','train.csv (40000 rows)',CODE,gpu,edges,runtime,parent_execution_succeeded=True)
    assert result and 'sm_86' in result[0] and 'sm_120' not in result[0]
    assert result[1]['calibration_file_sha256']==hashlib.sha256(file.read_bytes()).hexdigest()
    assert measured_optimizer('improve','train.csv (40000 rows)',CODE,LOCAL_GPU,edges,runtime,
                              parent_execution_succeeded=True) is None


def test_ROCm_workload_key_pins_HIP_and_gfx_without_transferring_CUDA_profile():
    node=next(n for n in HOSTS if n['id'].endswith('radeon_rx_9070_xt.spec'))
    gpu,edges,runtime=setup_host(node)
    key=workload_key(gpu,runtime,source_spec(CODE))
    assert key['hip']=='7.2' and key['gpu_architecture']=='gfx1201' and key['cuda'] is None
    assert architecture(gpu)=='gfx1201'


def test_driver_identical_memory_variants_keep_advice_but_disable_timing(monkeypatch,tmp_path):
    node=next(n for n in HOSTS if n['id'].endswith('geforce_rtx_4070_gddr6.spec'))
    gpu,edges,runtime=setup_host(node)
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_PATH',str(ROOT/'hwkg/graph.json'))
    monkeypatch.setenv('MLEVOLVE_HWKG_GRAPH_SHA256',SHA)
    monkeypatch.setenv('MLEVOLVE_HWKG_HARDWARE_ID',node['id'].removeprefix('hw:'))
    monkeypatch.setenv('MLEVOLVE_HWKG_EVIDENCE_POLICY','v8')
    monkeypatch.setenv('MLEVOLVE_HWKG_AUDIT_DIR',str(tmp_path))
    monkeypatch.setattr(injection,'_visible_cuda_device',lambda:gpu)
    monkeypatch.setattr('hwkg.host_policy.runtime_facts',lambda:runtime)
    monkeypatch.setattr('hwkg.calibration.calibration_flags',lambda:{})
    text=injection.hardware_guidance('draft','CNN classification')
    audit=json.loads((tmp_path/'prompt-audit.jsonl').read_text())
    assert text and audit['runtime_facts']['hardware_profile_identity_verified'] is False
    assert len(audit['runtime_facts']['matching_graph_profile_ids'])==2
    assert audit['calibration_trial'] is None
    assert select_record(calibration_path(LOCAL_GPU),LOCAL_GPU,
                         {**MEASURED_RUNTIME,'hardware_profile_identity_verified':False},source_spec(CODE))[0] is None


def test_ROCm_owned_precision_record_matches_architecture_and_HIP(tmp_path):
    node=next(n for n in HOSTS if n['id'].endswith('radeon_rx_9070_xt.spec'))
    gpu,edges,runtime=setup_host(node)
    code='\n'.join(line for line in CODE.splitlines() if not line.startswith('torch.backends.'))
    spec=source_spec(code)
    key=workload_key(gpu,runtime,spec)
    doc={'schema':'hwkg-calibration-v1','records':[{'workload':key,'quality_verified':False,
        'profiles':{'bf16_channels_last':{'step_seconds':.01,'setup_seconds':0.,'repeats':3,
        'normal_updates':25,'finite_updates':True,'finite_check_scope':'all_losses_initial_grad_final_parameters'}}}]}
    path=tmp_path/'rocm-profile.json';path.write_text(json.dumps(doc))
    assert select_record(path,gpu,runtime,spec)[0]
    assert select_record(path,gpu,{**runtime,'hip':'7.1'},spec)[0] is None
    other=NS(**{**vars(gpu),'gcnArchName':'gfx1100'})
    assert select_record(path,other,runtime,spec)[0] is None


@pytest.mark.parametrize('data',[[],{}, {'schema':'hwkg-optimizer-calibration-v1','quality_verified':False,
    'full_pipeline_gain_verified':False,'host':{},'workload':{},'runtime_flags':{},'profiles':{} }])
def test_external_optimizer_registry_rejects_malformed_records(tmp_path,monkeypatch,data):
    path=tmp_path/'invalid.json';path.write_text(json.dumps(data))
    monkeypatch.setenv('MLEVOLVE_HWKG_OPTIMIZER_CALIBRATION_PATH',str(path))
    assert load_record() is None
