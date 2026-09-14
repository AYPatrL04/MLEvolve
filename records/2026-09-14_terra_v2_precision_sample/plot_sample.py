"""Combined turnaround Gantt and sample-node metrics; no GPU-quality claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
manifest = json.loads((OUT / 'manifest.json').read_text())
checks = json.loads((OUT / 'checks.json').read_text())
outcome = json.loads((OUT / 'preflight.json').read_text())
raw = json.loads(Path(outcome['report_path']).read_text())
origin = manifest['prompt_ready_unix']
phases = [
    ('Terra response turnaround', origin, manifest['response_saved_unix'], '#3975B9'),
    ('CPU preflight', checks['preflight_started_unix'], checks['preflight_finished_unix'], '#D18A25'),
    ('CPU dtype / update smoke', checks['cpu_smoke_started_unix'], checks['cpu_smoke_finished_unix'], '#31946C'),
]
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
ax = fig.add_subplot(grid[0, :])
for index, (label, start, end, color) in enumerate(phases):
    ax.barh(index, end - start, left=start - origin, height=.52, color=color)
    ax.text(end - origin + 2, index, f'{end-start:.1f}s', va='center', fontsize=10)
ax.set_yticks(range(len(phases)), [p[0] for p in phases])
ax.invert_yaxis()
ax.set_xlim(0, max(p[2] for p in phases) - origin + 25)
ax.set_xlabel('Seconds since fresh prompt was saved (includes dispatch and audit gaps)')
ax.set_title('Fresh Terra sample — v2 curated HWDB, V100, normal mode', loc='left', fontweight='bold')
ax.grid(axis='x', alpha=.2)
ax.legend(handles=[Patch(color=p[3], label=p[0]) for p in phases], loc='upper right', fontsize=9)
left = fig.add_subplot(grid[1, 0])
metrics = [
    ('No whole-model half cast', not checks['whole_model_half_cast_lines']),
    ('FP32 parameters and state', checks['parameter_dtypes'] == ['torch.float32'] and checks['floating_optimizer_state_dtypes'] == ['torch.float32']),
    ('Finite CPU gradients / loss', checks['all_gradients_finite'] and checks['cpu_training_loss_finite']),
    ('Optimizer updates parameters', checks['optimizer_updates_parameters']),
    ('Precision validator', not checks['precision_issues']),
    ('Preflight admission', checks['preflight_admitted']),
]
for y, (label, value) in enumerate(metrics):
    left.scatter(1, y, color='#31946C' if value else '#C64747', s=95, marker='o' if value else 'X')
    left.text(1.08, y, 'PASS' if value else 'FAIL', va='center', fontsize=9)
left.set_yticks(range(len(metrics)), [m[0] for m in metrics])
left.set_xticks([1], ['Sample node 1'])
left.set_xlim(.8, 1.5)
left.invert_yaxis()
left.set_title('Audit checks by sample node (CPU / static)')
left.grid(alpha=.15)
right = fig.add_subplot(grid[1, 1])
loss = next(s for s in raw['stages'] if s['name'] == 'cpu_training')['evidence']['scenarios'][0]['loss']
right.scatter([1], [loss], color='#3975B9', s=85)
right.annotate(f'{loss:.5f}', (1, loss), xytext=(12, 8), textcoords='offset points')
right.set_xlim(.5, 1.5)
right.set_ylim(0, loss * 1.8)
right.set_xticks([1], ['Sample node 1'])
right.set_ylabel('Synthetic CPU fixture MSE')
right.set_title('Smoke loss by sample node — not task quality')
right.text(.03, .97, 'Task RMSE: not measured\nGPU epoch time: not measured\nPreflight rejected the RMSE API call', transform=right.transAxes, va='top', fontsize=10)
right.grid(alpha=.15)
fig.savefig(OUT / 'audit.png', dpi=120)
plt.close(fig)
