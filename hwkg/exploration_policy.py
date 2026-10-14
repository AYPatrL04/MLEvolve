"""Capability-guided candidates for this host, distinct from measured projections.

Advice only: no candidate/source/reward/API modifications or extra benchmark work.
Exact calibration remains optional evidence for claiming a measured projection.
"""
import ast
import re
from .adaptive_policy import source_configured
from .source_analysis import precision_scope


def optimizer_option(code):
    """One literal CUDA AdamW call; groups do not borrow a timing record."""
    try: tree=ast.parse(code)
    except SyntaxError: return None
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
           and ast.unparse(n.func)=='torch.optim.AdamW']
    if len(calls)!=1: return None
    if any(isinstance(n,ast.Call) and getattr(n.func,'id',getattr(n.func,'attr','')) in
           {'SAM','first_step','second_step'} for n in ast.walk(tree)): return None
    options={'fused':None,'foreach':None,'differentiable':False,'capturable':False}
    try:
        for k in calls[0].keywords:
            if k.arg in options: options[k.arg]=ast.literal_eval(k.value)
    except (ValueError,TypeError): return None
    if (any(type(v) is not bool and v is not None for v in options.values())
            or options['foreach'] is True or options['differentiable'] is True
            or options['capturable'] is True): return None
    return options


def guidance(stage, code, device, edges, runtime, *, phase, parent_execution_succeeded,
             blocked_profiles=(), identity_already_present=False, max_chars=750):
    if (device.name!='NVIDIA GeForce RTX 5080' or (device.major,device.minor)!=(12,0)
            or phase=='merge' or stage=='debug'): return None
    def supported(fid):
        p=edges.get(fid,{}).get('properties',{})
        return p.get('verified') is True and p.get('support_level')=='supported'
    bf16=supported('feat:amp') and supported('feat:bf16') and runtime.get('bf16_supported') is True
    fused=supported('feat:optimizer_fused_adamw') and runtime.get('adamw_fused_api') is True
    if not code:
        if stage!='draft' or not bf16: return None
        header='HWKG v8.16 sm_120 options:' if identity_already_present else 'HWKG v8.16 RTX5080 sm_120 options:'
        lines=[header,'[precision_choice] Consider TRAIN BF16 autocast; keep FP32 params/eval dtype. FP16 needs GradScaler.']
        ids=['precision_choice'];features=['feat:amp','feat:bf16']
        if fused:
            lines.append('[optimizer_choice] CUDA AdamW may try fused=True with FP32 params/BF16.')
            ids.append('optimizer_choice');features.append('feat:optimizer_fused_adamw')
        lines.append('Unmeasured speed/quality; preserve budget/outputs, no extra passes.')
        text='\n'.join(lines)
        if len(text)>min(max_chars,340): return None
        return text, {'advice_ids':ids,'selected_feature_ids':features,
                      'draft_recipe_forced':False,'draft_identity_facts_deduplicated':identity_already_present,
                      'stage_role':phase,'context_budget':min(max_chars,340),
                      'proposal_kind':'capability_guided_unmeasured','calibration_trial':None,
                      'measurements_verified':False,'end_to_end_gain_verified':False,
                      'declared_or_measured_cost_gate_applied':False,'added_provider_calls':0}
    if parent_execution_succeeded is not True: return None
    # Do not make hardware suggestions for an observed CPU-only implementation.
    if not re.search(r'\btorch\b',code) or not re.search(r'cuda|\.to\(.*DEV|\.to\(.*device',code,re.I): return None
    scope=precision_scope(code);types=scope.get('training_dtypes',[])
    configured=source_configured(code);opt=optimizer_option(code)
    ids=[];features=[];body=None;profile=None
    if (bf16 and fused and opt and opt['fused'] in (None,False) and types==['bfloat16']
            and not scope.get('training_scaler') and 'optimizer_fused_adamw' not in blocked_profiles
            and not re.search(r'\.half\s*\(|set_default_dtype',code)):
        ids=['optimizer_fused_adamw'];features=['feat:optimizer_fused_adamw'];profile='optimizer_fused_adamw'
        body='Try only fused=True on existing CUDA AdamW with FP32 parameters/BF16; retain groups/LR/WD. Check finite ordinary updates. Speed/quality unmeasured.'
    elif (bf16 and types==['bfloat16'] and opt and opt['fused'] in (None,False)
            and runtime.get('grad_scaler_api') is True
            and 'fp16_'+('channels_last' if configured.get('channels_last') else 'contiguous') not in blocked_profiles):
        ids=['precision'];features=['feat:amp'];profile='fp16_'+('channels_last' if configured.get('channels_last') else 'contiguous')
        body='Trial TRAIN-only torch.float16 autocast with GradScaler on FP32 parameters; unscale before existing clipping. Keep eval dtype separate and AdamW unfused. Track successful finite updates; retain BF16 if unstable. Speed/quality unmeasured.'
    elif (bf16 and types==['float16'] and opt
            and 'bf16_'+('channels_last' if configured.get('channels_last') else 'contiguous') not in blocked_profiles
            and not re.search(r'\.half\s*\(|set_default_dtype',code)):
        ids=['precision'];features=['feat:amp','feat:bf16'];profile='bf16_'+('channels_last' if configured.get('channels_last') else 'contiguous')
        body='Trial TRAIN-only torch.bfloat16 autocast with FP32 parameters and ordinary backward/step; disable TRAIN GradScaler. Preserve shared eval precision and optimizer options. Check finite ordinary updates. Candidate speed/quality unverified.'
    if not body: return None
    text=(f'HWKG v8.16 sm_120 {phase}: capability-guided hypothesis, no gain projection.\n'
          'Preserve batch/data order/epochs/recipe/eval/checkpoint and original response format. '
          'Use ordinary budgeted training; no extra benchmark or BN pass.\n'
          f'[{ids[0]}] {body}')
    if len(text)>min(max_chars,750): return None
    return text,{'advice_ids':ids,'selected_feature_ids':features,'stage_role':phase,
                 'proposal_kind':'capability_guided_unmeasured','calibration_trial':None,
                 'exploration_trial':{'profile':profile,'change':'optimizer_implementation' if ids[0]=='optimizer_fused_adamw' else 'training_precision',
                                      'quality_verified':False,'end_to_end_verified':False},
                 'context_budget':min(max_chars,750),'measurements_verified':False,
                 'end_to_end_gain_verified':False,'declared_or_measured_cost_gate_applied':False,
                 'added_provider_calls':0}
