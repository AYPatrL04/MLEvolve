import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from utils.muon_optimizer import MuonAdamW, validate_optimizer_coverage
from utils.training_diagnostics import TrainingDiagnostics


def test_fp32_update_matches_native_muon_with_only_precision_changed(monkeypatch):
    native = pytest.importorskip('torch.optim._muon')
    source = inspect.getsource(native._zeropower_via_newtonschulz)
    assert source.count('grad.bfloat16()') == 1
    scope = dict(native.__dict__)
    exec(source.replace('grad.bfloat16()', 'grad.to(dtype=torch.float32, copy=True)'), scope)
    monkeypatch.setattr(native, '_zeropower_via_newtonschulz', scope['_zeropower_via_newtonschulz'])
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    reference = copy.deepcopy(model)
    opt = MuonAdamW(model, hidden_names=['0.weight'])
    muon = torch.optim.Muon([reference[0].weight], lr=1e-3, weight_decay=1e-4,
                            momentum=0.95, ns_steps=5, adjust_lr_fn='match_rms_adamw')
    adam = torch.optim.AdamW([p for n, p in reference.named_parameters() if n != '0.weight'],
                             lr=1e-3, weight_decay=1e-4)
    for _ in range(4):
        for p, q in zip(model.parameters(), reference.parameters()):
            grad = torch.randn_like(p)
            p.grad = grad.clone()
            q.grad = grad.clone()
        opt.step()
        muon.step()
        adam.step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_missing_hidden_parameters_rejected_before_training():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    opt = torch.optim.AdamW(model[1].parameters())
    with pytest.raises(ValueError, match='missing'):
        TrainingDiagnostics(model, opt)
    with pytest.raises(TypeError, match='not a list'):
        validate_optimizer_coverage(model, [opt])


def test_hybrid_updates_all_parameters_and_resumes_under_autocast():
    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 1))
    opt = MuonAdamW(model, hidden_names=['0.weight'])
    initial = copy.deepcopy(model.state_dict())
    x = torch.randn(8, 4)
    model(x).square().mean().backward()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        opt.step()
    opt.zero_grad(set_to_none=True)
    assert all(not torch.equal(p, initial[n]) for n, p in model.named_parameters())
    assert len(opt.state) == 4
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    scheduler.step()
    other = copy.deepcopy(model)
    resumed = MuonAdamW(other, hidden_names=['0.weight'])
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    for net, optimizer in [(model, opt), (other, resumed)]:
        net(x).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), other.parameters()))


@pytest.mark.parametrize('mode', ['conservative', 'normal'])
def test_muon_parameters_survive_precision_filter_and_budget(mode):
    from agents.design_knowledge import hardware_records
    from knowledge.records import from_source, render_records
    graph = json.loads(Path('schema/hardware_knowledge_graph.json').read_text())
    node = next(n for n in graph['nodes'] if n['id'] == 'feat:muon_optimizer')
    records = from_source(node['properties'], domain='hardware')
    context = SimpleNamespace(candidate={}, raw_context={'design_records_v2': records},
        compact_context={'precision_policy': {'mode': mode, 'allowed_policies': ['fp32', 'disabled'] if mode == 'conservative' else ['fp32', 'fp16_amp', 'disabled']}})
    prompt = render_records(hardware_records(context), max_chars=3500)
    for text in ['MuonAdamW', 'momentum=0.95', 'ns_steps=5', 'lr=1e-3', 'docs.pytorch.org']:
        assert text in prompt
    assert len(prompt) <= 3500
