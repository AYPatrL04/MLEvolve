"""Single-optimizer FP32 Muon/AdamW integration.

Algorithm and starting settings: PyTorch 2.12 torch.optim.Muon documentation.
Unlike native Muon, orthogonalization stays FP32 for strict precision policies.
"""
import math

import torch


def validate_optimizer_coverage(model, optimizer):
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError('Return one torch.optim.Optimizer, not a list of optimizers')
    expected = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    owned = [id(p) for group in optimizer.param_groups for p in group['params']]
    missing = [name for identity, name in expected.items() if identity not in owned]
    if missing or len(owned) != len(set(owned)):
        raise ValueError(f'Optimizer coverage error: missing={missing}; duplicate parameters={len(owned) != len(set(owned))}')


class MuonAdamW(torch.optim.Optimizer):
    """Muon for explicitly named hidden matrices, AdamW for every other tensor."""
    def __init__(self, model, *, hidden_names, lr=1e-3, weight_decay=1e-4,
                 muon_lr=None, momentum=0.95, ns_steps=5):
        if lr < 0 or weight_decay < 0 or (muon_lr is not None and muon_lr < 0):
            raise ValueError('Learning rates and weight decay must be nonnegative')
        if not 0 <= momentum < 1 or not isinstance(ns_steps, int) or not 1 <= ns_steps < 100:
            raise ValueError('Invalid momentum or Newton-Schulz steps')
        names = set(hidden_names)
        named = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if not names or not names <= named.keys():
            raise ValueError('Specify existing trainable hidden matrix names explicitly')
        if any(named[n].ndim != 2 for n in names):
            raise ValueError('Muon accepts only 2D hidden weights')
        if any(p.dtype != torch.float32 for p in named.values()):
            raise ValueError('MuonAdamW requires FP32 parameters')
        groups = [dict(params=[p for n, p in named.items() if n in names],
                       lr=lr if muon_lr is None else muon_lr, weight_decay=weight_decay,
                       momentum=momentum, ns_steps=ns_steps, optimizer_kind='muon_fp32'),
                  dict(params=[p for n, p in named.items() if n not in names], lr=lr,
                       weight_decay=weight_decay, optimizer_kind='adamw')]
        if not groups[1]['params']:
            raise ValueError('Keep output heads and non-hidden parameters on AdamW')
        self.adam = torch.optim.AdamW(groups[1]['params'], lr=lr, weight_decay=weight_decay)
        groups[1].update(self.adam.param_groups[0])
        super().__init__(groups, dict(lr=lr))
        self._bind()
        validate_optimizer_coverage(model, self)

    def _bind(self):
        self.adam.param_groups = [self.param_groups[1]]
        self.adam.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        # Disable ambient autocast too: FP32 parameters alone do not guarantee FP32 NS.
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse or p.grad.dtype != torch.float32:
                raise ValueError('Muon needs dense FP32 gradients')
            with torch.autocast(device_type=p.device.type, enabled=False):
                state = self.state[p]
                if not state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.lerp_(p.grad, 1-group['momentum'])
                update = p.grad.lerp(buf, group['momentum'])
                transposed = update.shape[0] > update.shape[1]
                if transposed:
                    update = update.T
                update.div_(update.norm().clamp(min=1e-7))
                for _ in range(group['ns_steps']):
                    gram = update @ update.T
                    polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
                    update = torch.addmm(update, polynomial, update, beta=3.4445)
                if transposed:
                    update = update.T
                p.mul_(1-group['lr']*group['weight_decay'])
                p.add_(update, alpha=-group['lr']*0.2*math.sqrt(max(p.shape)))
        self._bind()
        self.adam.step()
        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._bind()
