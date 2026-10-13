"""Scoped syntax observations, never a claim that a path actually executed."""
import ast


def call_name(node):
    return getattr(node.func, 'attr', getattr(node.func, 'id', ''))


def scopes(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    # Module statements exclude function bodies, which have their own scope.
    module = ast.Module(body=[n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[])
    owners = {'<module>': module, **functions}
    direct = {k: list(ast.walk(n)) for k, n in owners.items()}
    training = {k for k, nodes in direct.items() if any(isinstance(n, ast.Call) and call_name(n) == 'backward' for n in nodes)}
    evaluation = {k for k, n in functions.items() if (
        any(word in k.lower() for word in ('predict', 'eval', 'valid', 'infer', 'test'))
        or any(isinstance(d, ast.Call) and call_name(d) in ('no_grad', 'inference_mode') for d in n.decorator_list))}
    def reachable(roots):
        found = set(roots)
        pending = list(roots)
        while pending:
            key = pending.pop()
            for n in direct[key]:
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions and n.func.id not in found:
                    found.add(n.func.id); pending.append(n.func.id)
        return found
    return tree, owners, direct, training, reachable(training), reachable(evaluation)


def precision_scope(code):
    parsed = scopes(code)
    if parsed is None:
        return {'training': None, 'evaluation': None, 'training_scaler': None, 'syntax_only': True}
    _, _, direct, _, train, evaluation = parsed
    def signature(keys):
        rows = []
        for key in sorted(keys):
            for n in direct[key]:
                if isinstance(n, ast.Call) and call_name(n) in ('autocast', 'half', 'bfloat16', 'GradScaler', 'set_autocast_dtype'):
                    # GradScaler is a training update mechanism, not eval precision.
                    if call_name(n) != 'GradScaler': rows.append(ast.dump(n, include_attributes=False))
        return sorted(rows)
    # Scalers are normally constructed outside train_epoch; observe construction
    # and usage separately. Disabled/unused constructors are not active evidence.
    scaler_names = {}
    for n in ast.walk(parsed[0]):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and call_name(n.value) == 'GradScaler':
            enabled = next((k.value for k in n.value.keywords if k.arg == 'enabled'), None)
            state = not (isinstance(enabled, ast.Constant) and enabled.value is False)
            for t in n.targets:
                if isinstance(t, ast.Name): scaler_names[t.id] = state
    used = {n.func.value.id for k in train for n in direct[k] if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
            and n.func.attr in ('scale', 'step', 'update')}
    active = [k for k in used if scaler_names.get(k) is True]
    dtype = set()
    for k in train:
        for n in direct[k]:
            if isinstance(n, ast.Call) and call_name(n) == 'autocast':
                for kw in n.keywords:
                    if kw.arg == 'dtype' and isinstance(kw.value, ast.Attribute): dtype.add(kw.value.attr)
    return {'training': signature(train), 'evaluation': signature(evaluation),
            'training_dtypes': sorted(dtype), 'training_scaler': sorted(active),
            'bf16_with_active_scaler': dtype == {'bfloat16'} and bool(active), 'syntax_only': True}


def training_scope_uncertain(code):
    parsed = scopes(code)
    if parsed is None: return True
    tree, owners, direct, roots, _, _ = parsed
    if not roots: return True
    # Freezing remains conservative. Logging detach and eval-only no_grad no
    # longer poison an otherwise direct full-model training recipe.
    if any(isinstance(n, ast.Call) and call_name(n) == 'requires_grad_' for n in ast.walk(tree)): return True
    if any(isinstance(n, ast.Attribute) and n.attr == 'requires_grad' and isinstance(n.ctx, ast.Store) for n in ast.walk(tree)): return True
    for key in roots:
        root = owners[key]
        if isinstance(root, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                isinstance(d, ast.Call) and call_name(d) in ('no_grad', 'inference_mode') for d in root.decorator_list): return True
        for n in direct[key]:
            if isinstance(n, ast.With) and any(isinstance(i.context_expr, ast.Call) and call_name(i.context_expr) in ('no_grad', 'inference_mode') for i in n.items):
                # A no_grad forward/loss in the backward scope invalidates scope;
                # metrics/checkpoint operations on detached tensors do not.
                if any(isinstance(c, ast.Call) and (call_name(c) in ('cross_entropy', 'CrossEntropyLoss')
                       or isinstance(c.func, ast.Name) and c.func.id in ('model', 'net')) for s in n.body for c in ast.walk(s)): return True
            if isinstance(n, ast.Assign) and any(isinstance(c, ast.Call) and call_name(c) == 'detach' for c in ast.walk(n.value)):
                targets = {t.id for t in n.targets if isinstance(t, ast.Name)}
                if targets & {'loss', 'logits', 'outputs', 'out', 'pred', 'prediction'}: return True
    return False
