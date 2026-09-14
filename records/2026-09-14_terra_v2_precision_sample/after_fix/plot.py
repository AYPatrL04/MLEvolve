"""Compare the archived sample and its local correction without quality claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parent
old = json.loads((OUT.parent / 'manifest.json').read_text())
checks = json.loads((OUT.parent / 'checks.json').read_text())
fixed = json.loads((OUT / 'result.json').read_text())
colors = ['#3975B9', '#D18A25', '#31946C', '#8D63B8']
labels = ['Terra response turnaround', 'CPU preflight', 'CPU safety checks', 'Synthetic CPU workflow']
cases = [
    [old['response_turnaround_seconds'], checks['preflight_finished_unix'] - checks['preflight_started_unix'],
     checks['cpu_smoke_finished_unix'] - checks['cpu_smoke_started_unix'], 0],
    [0, fixed['preflight_finished'] - fixed['preflight_started'],
     fixed['fallback_finished'] - fixed['preflight_finished'], fixed['execution_finished'] - fixed['fallback_finished']],
]
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
grid = fig.add_gridspec(2, 2)
ax = fig.add_subplot(grid[0, :])
for row, times in enumerate(cases):
    start = 0
    for duration, color in zip(times, colors):
        if duration:
            ax.barh(row, duration, left=start, height=.45, color=color)
        start += duration
    ax.text(start + 2, row, f'{start:.1f}s recorded work', va='center')
ax.set_yticks([0, 1], ['Original Terra sample', 'Local correction'])
ax.invert_yaxis()
ax.set_xlim(0, 150)
ax.set_xlabel('Seconds per case; local correction has no new LLM generation')
ax.set_title('Current v2 HWDB audit — original sample and verified correction', loc='left', weight='bold')
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2)
ax.grid(axis='x', alpha=.2)
left = fig.add_subplot(grid[1, 0])
left.plot([1, 2], [0, 1], color='#888888', linestyle=':')
left.scatter([1, 2], [0, 1], c=['#C64747', '#31946C'], s=100)
left.set_xticks([1, 2], ['Original', 'Corrected'])
left.set_yticks([0, 1], ['Rejected', 'CPU admitted'])
left.set_ylim(-.35, 1.5)
left.set_title('Admission result by sample node')
left.annotate('RMSE API incompatible', (1, 0), xytext=(8, 10), textcoords='offset points')
left.annotate('INCONCLUSIVE: GPU check remains', (2, 1), xytext=(-235, 20), textcoords='offset points')
left.grid(alpha=.2)
right = fig.add_subplot(grid[1, 1])
right.scatter([2], [fixed['synthetic_rmse']], color=colors[3], s=100)
right.set_xticks([1, 2], ['Original', 'Corrected'])
right.set_xlim(.5, 2.5)
right.set_ylim(0, 65)
right.set_ylabel('Synthetic holdout RMSE')
right.set_title('Fixture metric by sample node — not real task quality')
right.text(1, 10, 'Not executed', ha='center', color='#666666')
right.annotate(f"{fixed['synthetic_rmse']:.4f}", (2, fixed['synthetic_rmse']), xytext=(8, 10), textcoords='offset points')
right.text(.03, .97, 'Selected precision: FP32\nQuality evidence: unverified\nNo GPU training or epoch-speed comparison', transform=right.transAxes, va='top')
right.grid(alpha=.2)
fig.savefig(OUT / 'comparison.png', dpi=120)
plt.close(fig)
