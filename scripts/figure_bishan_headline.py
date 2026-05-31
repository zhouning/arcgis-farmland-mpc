"""
Bishan headline figure for CommsEE submission.

Two panels:
  (a) per-step trajectories of slope reduction across 5 contrastive-MPC episodes,
      with PPO/MARL endpoint markers and the 180-s SA matched-budget endpoint.
  (b) county-scale map of 852 farm/forest swap locations on the optimised cadastre.

Inputs (regime B/C, the public-package reproduction):
  - /Users/zhouning/farmland_mpc_runs/bishan/mpc_output/mpc_run.log         (per-step trajectory)
  - /Users/zhouning/farmland_mpc_runs/bishan/mpc_output/optimized.shp        (CHG_FLAG geometry)
  - in-house DRL summaries from main-text §5

Outputs:
  - paper/figures_v2/bishan_headline.pdf
  - paper/figures_v2/bishan_headline.png

Usage (inside farmland-mpc conda env):
  python scripts/figure_bishan_headline.py
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd

# Paths
ROOT = Path(__file__).resolve().parents[1]
LOG = Path('/Users/zhouning/farmland_mpc_runs/bishan/mpc_output/mpc_run.log')
SHP = Path('/Users/zhouning/farmland_mpc_runs/bishan/mpc_output/optimized.shp')
OUT_PDF = ROOT / 'paper/figures_v2/bishan_headline.pdf'
OUT_PNG = ROOT / 'paper/figures_v2/bishan_headline.png'

# Style: nature-style soft tones, single-column friendly
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.6,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'pdf.fonttype': 42,   # editable in Illustrator
    'ps.fonttype': 42,
})

# ---- (a) parse trajectory ----
trajs: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
cur_ep = None
for line in LOG.read_text().splitlines():
    m = re.search(r'Episode (\d+)/5', line)
    if m:
        cur_ep = int(m.group(1)) - 1
        continue
    m = re.search(r'step\s+(\d+)/100 slope=([-+\d\.]+)% cont=([-+\d\.]+) baimu_ha=([-+\d\.]+)', line)
    if m and cur_ep is not None:
        trajs[cur_ep].append((int(m.group(1)),
                              float(m.group(2)),
                              float(m.group(3)),
                              float(m.group(4))))
# Add (step 0, slope 0) as initial state for each episode
for ep in trajs:
    trajs[ep].insert(0, (0, 0.0, 0.0, 0.0))

# Baselines (means from main-text §5)
PPO_MEAN, PPO_SD = -0.79, 0.36
MARL_MEAN, MARL_SD = -0.81, 0.10
SA_180S_REWARD = 9.04   # for caption only
MPC_180S_REWARD = 86.74

# ---- figure layout ----
fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=False)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.28,
                      left=0.075, right=0.985, top=0.92, bottom=0.13)
ax_traj = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[0, 1])

# ===== panel (a): trajectories =====
# Aggregate across 5 episodes: mean line + min/max band
all_steps = sorted({step for pts in trajs.values() for (step, *_rest) in pts})
slopes_by_step = {step: [] for step in all_steps}
for pts in trajs.values():
    for step, slope, _, _ in pts:
        slopes_by_step[step].append(slope)
xs = np.array(all_steps)
mean_y = np.array([np.mean(slopes_by_step[s]) for s in all_steps])
min_y = np.array([np.min(slopes_by_step[s]) for s in all_steps])
max_y = np.array([np.max(slopes_by_step[s]) for s in all_steps])

# 5-episode envelope: thin shaded band
ax_traj.fill_between(xs, min_y, max_y, color='#1f77b4', alpha=0.18,
                     linewidth=0, label='5-episode envelope')
# Mean trajectory in bold
ax_traj.plot(xs, mean_y, color='#1f77b4', linewidth=1.4,
             marker='o', markersize=3.0, markeredgewidth=0,
             label='Contrastive MPC (mean of 5 episodes)')

# Final-step horizontal lines for PPO and MARL means
ax_traj.axhline(PPO_MEAN, linestyle='--', color='#d95f02', linewidth=0.9,
                alpha=0.9, label=f'Centralised PPO (−0.79 ± 0.36%)')
ax_traj.axhline(MARL_MEAN, linestyle=':', color='#7570b3', linewidth=0.9,
                alpha=0.9, label=f'MARL (−0.81 ± 0.10%)')
# 1-σ shaded bands for the baselines, light tone, no legend entry
ax_traj.fill_between([0, 100], PPO_MEAN - PPO_SD, PPO_MEAN + PPO_SD,
                     color='#d95f02', alpha=0.07, linewidth=0)
ax_traj.fill_between([0, 100], MARL_MEAN - MARL_SD, MARL_MEAN + MARL_SD,
                     color='#7570b3', alpha=0.10, linewidth=0)
# Endpoint annotation at right edge for the contrastive band only
mean_final = mean_y[-1]
ax_traj.annotate(f'−2.04%\n(env-side)\n−2.001% (verification)',
                 xy=(100, mean_final), xytext=(105, mean_final - 0.05),
                 ha='left', va='center', fontsize=6.5,
                 color='#1f77b4', linespacing=1.2,
                 arrowprops=dict(arrowstyle='-', color='#1f77b4',
                                 linewidth=0.4, alpha=0.5))

ax_traj.set_xlim(0, 132)
ax_traj.set_ylim(-2.30, 0.10)
ax_traj.set_xlabel('Planning step')
ax_traj.set_ylabel('Slope change (%, lower is better)')
ax_traj.set_title('a   Per-step slope-reduction trajectory on Bishan',
                  loc='left', pad=4)
ax_traj.grid(True, axis='y', linewidth=0.4, alpha=0.3)
for s in ('top', 'right'):
    ax_traj.spines[s].set_visible(False)
ax_traj.legend(loc='lower left', fontsize=6.5, frameon=False)

# ===== panel (b): swap-pattern map =====
shp = gpd.read_file(SHP)
unchanged = shp[shp['CHG_FLAG'] == 0]
farm2for = shp[shp['CHG_FLAG'] == 1]   # farm -> forest
for2farm = shp[shp['CHG_FLAG'] == 2]   # forest -> farm

# County boundary outline via dissolve (faster than polygon-by-polygon)
county = unchanged.geometry.unary_union
county_gdf = gpd.GeoDataFrame(geometry=[county], crs=unchanged.crs)

# 1) Plot full unchanged background as a single tone (no per-parcel edges)
#    Rasterize the heavy parcel layer to keep the PDF small (vector-text retained).
unchanged.plot(ax=ax_map, facecolor='#e8e8e8', edgecolor='none', linewidth=0,
               rasterized=True)
# 2) County boundary thin outline (kept as vector)
county_gdf.boundary.plot(ax=ax_map, color='#888888', linewidth=0.4)
# 3) Buffer swap parcels so they are visible at county scale
buf_radius = 250
farm2for_buf = farm2for.copy(); farm2for_buf['geometry'] = farm2for.buffer(buf_radius)
for2farm_buf = for2farm.copy(); for2farm_buf['geometry'] = for2farm.buffer(buf_radius)
farm2for_buf.plot(ax=ax_map, facecolor='#d62728', edgecolor='none',
                  linewidth=0, alpha=0.85, rasterized=True)
for2farm_buf.plot(ax=ax_map, facecolor='#2ca02c', edgecolor='none',
                  linewidth=0, alpha=0.85, rasterized=True)

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='#d62728', edgecolor='none',
          label=f'Farm $\\to$ Forest ({len(farm2for)} parcels)'),
    Patch(facecolor='#2ca02c', edgecolor='none',
          label=f'Forest $\\to$ Farm ({len(for2farm)} parcels)'),
    Patch(facecolor='#e8e8e8', edgecolor='#888888',
          linewidth=0.4,
          label=f'Unchanged ({len(unchanged):,} parcels)'),
]
ax_map.set_aspect('equal')
ax_map.set_xticks([])
ax_map.set_yticks([])
for s in ('top', 'right', 'bottom', 'left'):
    ax_map.spines[s].set_visible(False)
ax_map.set_title('b   Optimised swap pattern (Bishan, episode 0; '
                 '852 paired swaps)',
                 loc='left', pad=4)

# Tighten map limits to actual data extent (avoid white margins)
xmin, ymin, xmax, ymax = shp.total_bounds
xpad = (xmax - xmin) * 0.02
ypad = (ymax - ymin) * 0.02
ax_map.set_xlim(xmin - xpad, xmax + xpad)
ax_map.set_ylim(ymin - ypad, ymax + ypad)

# Scale bar at top-right
bar_x0 = xmax - (xmax - xmin) * 0.22
bar_y0 = ymax - (ymax - ymin) * 0.04
ax_map.plot([bar_x0, bar_x0 + 10000], [bar_y0, bar_y0],
            color='black', linewidth=1.5, solid_capstyle='butt')
ax_map.text(bar_x0 + 5000, bar_y0 - (ymax - ymin) * 0.012,
            '10 km', ha='center', va='top', fontsize=7)

# North arrow at top-left of map
arrow_x0 = xmin + (xmax - xmin) * 0.08
arrow_y0 = ymax - (ymax - ymin) * 0.04
ax_map.annotate('', xy=(arrow_x0, arrow_y0),
                xytext=(arrow_x0, arrow_y0 - (ymax - ymin) * 0.06),
                arrowprops=dict(arrowstyle='-|>', color='black',
                                lw=0.8, mutation_scale=8))
ax_map.text(arrow_x0, arrow_y0 - (ymax - ymin) * 0.075,
            'N', ha='center', va='top', fontsize=8, fontweight='bold')

# Legend at bottom-left of the map
ax_map.legend(handles=legend_handles, loc='lower left',
              fontsize=6.5, frameon=False,
              bbox_to_anchor=(0.0, 0.0))

# Save both PDF (vector) and PNG (preview)
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, format='pdf', bbox_inches='tight', pad_inches=0.04, dpi=300)
fig.savefig(OUT_PNG, format='png', dpi=300, bbox_inches='tight', pad_inches=0.04)
print(f'wrote {OUT_PDF}')
print(f'wrote {OUT_PNG}')
