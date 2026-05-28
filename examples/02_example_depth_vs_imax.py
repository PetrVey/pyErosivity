# -*- coding: utf-8 -*-
"""
Compares IMax30-only vs dual-threshold (IMax30 OR depth) erosivity
event selection across five temporal resolutions (5, 10, 15, 30, 60 min).

The full RUSLE definition (Wischmeier & Smith 1978; Renard et al. 1997)
flags an event as erosive if EITHER:
    (i)  IMax30            >= intensity_threshold [mm/h]
    (ii) accumulated depth >= accum_threshold     [mm]

Data are resampled from the base 5-min record. For 60-min resolution
IMax30 = IMax60 (single hourly reading; Williams & Sheridan 1991), so
the accumulation window is set to 60 min. Thresholds are kept constant
across resolutions to show how aggregation shifts events between zones.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyErosivity import remove_incomplete_years
from pyErosivity import get_events
from pyErosivity import get_events_values
from pyErosivity import compute_erosivity
from pyErosivity import get_only_erosivity_events
from pyErosivity import get_mean_annual_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, '..', 'res')
_OUT = os.path.join(_HERE, '..', 'out')
_FIG = os.path.join(_HERE, '..', 'fig')

# %%
save_results = True
station_num = "VE_0091"
slice_year_from = "1990"
slice_year_to = "2020"

# %%
# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
separation = 6          # Dry spell between events [hours]
                        # (Wischmeier and Smith, 1958, 1978)
min_rain = 0.1          # Minimum rain depth [mm]
name_col = "vals"       # Column with precipitation data
thr_imax30 = 12.7       # Wischmeier IMax30 threshold [mm/h]
thr_imax15 = 25.4       # Wischmeier IMax15 threshold [mm/h]
accum_threshold = 12.7  # Min accumulated event depth [mm]
                        # (Wischmeier & Smith 1978)

# Resolutions to analyse — resample rule, time step, imax column to use
# For 60-min: imax_60 because imax_30 cannot be resolved at hourly step
RESOLUTIONS = [
    {'label': '5 min',  'rule': None,    'tres': 5.0,  'imax_col': 'imax_30'},
    {'label': '15 min', 'rule': '15min', 'tres': 15.0, 'imax_col': 'imax_30'},
    {'label': '30 min', 'rule': '30min', 'tres': 30.0, 'imax_col': 'imax_30'},
    {'label': '60 min', 'rule': '60min', 'tres': 60.0, 'imax_col': 'imax_60'},
]

# IMax15-based resolutions — 5/15 min use imax_15; coarser use next window
RESOLUTIONS_15 = [
    {'label': '5 min',  'rule': None,    'tres': 5.0,  'imax_col': 'imax_15'},
    {'label': '15 min', 'rule': '15min', 'tres': 15.0, 'imax_col': 'imax_15'},
    {'label': '30 min', 'rule': '30min', 'tres': 30.0, 'imax_col': 'imax_30'},
    {'label': '60 min', 'rule': '60min', 'tres': 60.0, 'imax_col': 'imax_60'},
]

# IMax15 threshold applied to imax_30 column across all resolutions
# (equivalent to the old approach — kept for comparison)
RESOLUTIONS_15_I30 = [
    {'label': '5 min',  'rule': None,    'tres': 5.0,  'imax_col': 'imax_30'},
    {'label': '15 min', 'rule': '15min', 'tres': 15.0, 'imax_col': 'imax_30'},
    {'label': '30 min', 'rule': '30min', 'tres': 30.0, 'imax_col': 'imax_30'},
    {'label': '60 min', 'rule': '60min', 'tres': 60.0, 'imax_col': 'imax_60'},
]
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

# %%
# Load and preprocess 5-min base data once
data_base = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_5min_newflag.parguqet.gzip")
)
data_base['time'] = pd.to_datetime(data_base['time'])
data_base = data_base.set_index('time')
data_base = data_base.loc[slice_year_from:slice_year_to]

data_base.loc[data_base['flag'] > 0, name_col] = np.nan
data_base.loc[data_base[name_col] < min_rain, name_col] = 0

data_base, _ = remove_incomplete_years(
    data_base, name_col, nan_to_zero=True, tolerance=0.1
)

# %%
# Loop over resolutions, compute Venn zones for each
results = {}
table_rows = []
start_time = time.time()

all_years = sorted(
    pd.date_range(slice_year_from, slice_year_to, freq='YS').year
)


def mean_annual(df, years):
    return (
        df.groupby(df['event_start'].dt.year)
        .size()
        .reindex(years, fill_value=0)
        .mean()
    )


def annual_counts(df, years):
    return (
        df.groupby(df['event_start'].dt.year)
        .size()
        .reindex(years, fill_value=0)
        .values
    )


for res in RESOLUTIONS:
    label = res['label']
    tres = res['tres']
    imax_col = res['imax_col']

    if res['rule'] is None:
        df_res = data_base[[name_col]].copy()
    else:
        df_res = data_base[[name_col]].resample(res['rule']).sum()

    df_arr = np.array(df_res[name_col])
    df_dates = np.array(df_res.index)

    arr_dates = get_events(
        data=df_arr,
        dates=df_dates,
        separation=separation,
        min_rain=min_rain,
        check_gaps=False,
    )

    df_all = get_events_values(
        data=df_arr,
        dates=df_dates,
        arr_dates_oe=arr_dates,
        time_resolution=tres,
    )
    df_all = compute_erosivity(df_all)

    df_erosive = get_only_erosivity_events(
        df_all,
        imax_col=imax_col,
        intensity_threshold=thr_imax30,
        accum_threshold=accum_threshold,
        use_both_thresholds=True,
    )

    mask_i = df_erosive[imax_col] >= thr_imax30
    mask_d = df_erosive['event_depth'] >= accum_threshold

    results[label] = {
        'df_all':     df_all,
        'df_erosive': df_erosive,
        'imax_only':  df_erosive[mask_i & ~mask_d],
        'both':       df_erosive[mask_i & mask_d],
        'depth_only': df_erosive[~mask_i & mask_d],
        'none':       df_all[
            ~(df_all[imax_col] >= thr_imax30)
            & ~(df_all['event_depth'] >= accum_threshold)
        ],
        'imax_col':   imax_col,
    }

    m0 = mean_annual(results[label]['none'], all_years)
    m1 = mean_annual(results[label]['imax_only'], all_years)
    m12 = mean_annual(results[label]['both'], all_years)
    m2 = mean_annual(results[label]['depth_only'], all_years)

    r_factor = get_mean_annual_stats(
        df_erosive, all_years=all_years,
    )['erosivity']['mean']

    table_rows.append({
        'Resolution':        label,
        'Total ev/yr':       round(m0 + m1 + m12 + m2, 1),
        'Non-erosive ev/yr': round(m0, 1),
        'Erosive ev/yr':     round(m1 + m12 + m2, 1),
        'IMax only ev/yr':   round(m1, 1),
        'Both ev/yr':        round(m12, 1),
        'Depth only ev/yr':  round(m2, 1),
        'R-factor':          round(r_factor, 1),
    })


print(f"Total calculation time: {time.time() - start_time:.2f} s\n")
df_table = pd.DataFrame(table_rows).set_index('Resolution')
df_table.columns = [
    *df_table.columns[:-1],
    'R-factor\n[MJ mm ha-1 h-1 yr-1]',
]
print("Mean annual event counts by resolution and criterion zone:")
print(df_table.to_string())

print("\nTotal event counts (all years combined):")
for label, r in results.items():
    n_all = len(r['df_all'])
    n_erosive = len(r['df_erosive'])
    n1 = len(r['imax_only'])
    n12 = len(r['both'])
    n2 = len(r['depth_only'])
    print(
        f"  {label:15s} all={n_all:5d}  erosive={n_erosive:5d}"
        f"  imax_only={n1:4d}  both={n12:4d}  depth_only={n2:4d}"
    )

# %%
# == # == # == # == # == # == # == # == # == # ==
# == # == # PLOTS # == # == # PLOTS # == # == # ==
# == # == # == # == # == # == # == # == # == # ==

COLORS = {
    'none':       'lightgrey',
    'imax_only':  'steelblue',
    'both':       'mediumpurple',
    'depth_only': 'tomato',
}


# Figure 2: stacked bar — event counts by zone per resolution
labels_bar = list(results.keys())
n1_vals  = [len(results[l]['imax_only'])  for l in labels_bar]
n12_vals = [len(results[l]['both'])       for l in labels_bar]
n2_vals  = [len(results[l]['depth_only']) for l in labels_bar]
totals   = [a + b + c for a, b, c in zip(n1_vals, n12_vals, n2_vals)]

y = range(len(labels_bar))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle(
    f"{station_num} — Erosive event counts by zone",
    fontsize=13, fontweight='bold',
)

b1 = ax2.barh(y, n1_vals, color=COLORS['imax_only'], label='IMax only')
b2 = ax2.barh(
    y, n12_vals, color=COLORS['both'],
    left=n1_vals, label='Both',
)
b3 = ax2.barh(
    y, n2_vals, color=COLORS['depth_only'],
    left=[a + b for a, b in zip(n1_vals, n12_vals)],
    label='Depth only',
)

for i, (v1, v12, v2, tot) in enumerate(
    zip(n1_vals, n12_vals, n2_vals, totals)
):
    for val, left, color in [
        (v1,  0,        'white'),
        (v12, v1,       'white'),
        (v2,  v1 + v12, 'white'),
    ]:
        if val == 0:
            continue
        pct = val / tot * 100
        cx = left + val / 2
        ax2.text(
            cx, i, f'{val}\n({pct:.0f}%)',
            ha='center', va='center',
            fontsize=7, color=color, fontweight='bold',
        )

ax2.invert_yaxis()
ax2.set_yticks(list(y))
ax2.set_yticklabels(labels_bar)
ax2.set_xlabel('Total event count (all years)')
ax2.set_xlim(0, 1000)
ax2.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
    ncol=3, fontsize=9,
)
ax2.grid(axis='x', alpha=0.4)

plt.tight_layout()
if save_results:
    fig2.savefig(
        os.path.join(_FIG, '02_fig1.jpg'),
        format='jpeg', dpi=300,
    )
plt.show()

# %%
# == # == # == # == # == # == # == # == # == # ==
# == # IMax15 criterion (25.4 mm/h) analysis # ==
# == # == # == # == # == # == # == # == # == # ==

results_15 = {}

for res in RESOLUTIONS_15:
    label = res['label']
    tres = res['tres']
    imax_col = res['imax_col']

    if res['rule'] is None:
        df_res = data_base[[name_col]].copy()
    else:
        df_res = data_base[[name_col]].resample(res['rule']).sum()

    df_arr = np.array(df_res[name_col])
    df_dates = np.array(df_res.index)

    arr_dates = get_events(
        data=df_arr,
        dates=df_dates,
        separation=separation,
        min_rain=min_rain,
        check_gaps=False,
    )

    df_all = get_events_values(
        data=df_arr,
        dates=df_dates,
        arr_dates_oe=arr_dates,
        time_resolution=tres,
    )
    df_all = compute_erosivity(df_all)

    df_erosive = get_only_erosivity_events(
        df_all,
        imax_col=imax_col,
        intensity_threshold=thr_imax15,
        accum_threshold=accum_threshold,
        use_both_thresholds=True,
    )

    mask_i = df_erosive[imax_col] >= thr_imax15
    mask_d = df_erosive['event_depth'] >= accum_threshold

    results_15[label] = {
        'df_all':     df_all,
        'df_erosive': df_erosive,
        'imax_only':  df_erosive[mask_i & ~mask_d],
        'both':       df_erosive[mask_i & mask_d],
        'depth_only': df_erosive[~mask_i & mask_d],
        'none':       df_all[
            ~(df_all[imax_col] >= thr_imax15)
            & ~(df_all['event_depth'] >= accum_threshold)
        ],
        'imax_col':   imax_col,
    }

print("\nTotal event counts — IMax15 criterion (all years combined):")
for label, r in results_15.items():
    n_all = len(r['df_all'])
    n_erosive = len(r['df_erosive'])
    n1 = len(r['imax_only'])
    n12 = len(r['both'])
    n2 = len(r['depth_only'])
    print(
        f"  {label:15s} all={n_all:5d}  erosive={n_erosive:5d}"
        f"  imax_only={n1:4d}  both={n12:4d}  depth_only={n2:4d}"
    )


# Figure 4: stacked horizontal bar — IMax15 criterion
labels_bar15 = list(results_15.keys())
n1_v  = [len(results_15[l]['imax_only'])  for l in labels_bar15]
n12_v = [len(results_15[l]['both'])       for l in labels_bar15]
n2_v  = [len(results_15[l]['depth_only']) for l in labels_bar15]
totals15 = [a + b + c for a, b, c in zip(n1_v, n12_v, n2_v)]

y15 = range(len(labels_bar15))
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.suptitle(
    f"{station_num} — Erosive event counts by zone  "
    f"(IMax15 ≥ {thr_imax15} mm/h)",
    fontsize=13, fontweight='bold',
)

ax4.barh(y15, n1_v, color=COLORS['imax_only'], label='IMax only')
ax4.barh(
    y15, n12_v, color=COLORS['both'],
    left=n1_v, label='Both',
)
ax4.barh(
    y15, n2_v, color=COLORS['depth_only'],
    left=[a + b for a, b in zip(n1_v, n12_v)],
    label='Depth only',
)

for i, (v1, v12, v2, tot) in enumerate(
    zip(n1_v, n12_v, n2_v, totals15)
):
    for val, left in [
        (v1,  0),
        (v12, v1),
        (v2,  v1 + v12),
    ]:
        if val == 0:
            continue
        pct = val / tot * 100
        cx = left + val / 2
        ax4.text(
            cx, i, f'{val}\n({pct:.0f}%)',
            ha='center', va='center',
            fontsize=7, color='white', fontweight='bold',
        )

ax4.invert_yaxis()
ax4.set_yticks(list(y15))
ax4.set_yticklabels(labels_bar15)
ax4.set_xlabel('Total event count (all years)')
ax4.set_xlim(0, 1000)
ax4.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
    ncol=3, fontsize=9,
)
ax4.grid(axis='x', alpha=0.4)

plt.tight_layout()
if save_results:
    fig4.savefig(
        os.path.join(_FIG, '02_fig2.jpg'),
        format='jpeg', dpi=300,
    )
plt.show()

# %%
# == # == # == # == # == # == # == # == # == # ==
# == # IMax15 threshold on imax_30 column (old approach) # ==
# == # == # == # == # == # == # == # == # == # ==

results_15_i30 = {}

for res in RESOLUTIONS_15_I30:
    label = res['label']
    tres = res['tres']
    imax_col = res['imax_col']

    if res['rule'] is None:
        df_res = data_base[[name_col]].copy()
    else:
        df_res = data_base[[name_col]].resample(res['rule']).sum()

    df_arr = np.array(df_res[name_col])
    df_dates = np.array(df_res.index)

    arr_dates = get_events(
        data=df_arr,
        dates=df_dates,
        separation=separation,
        min_rain=min_rain,
        check_gaps=False,
    )

    df_all = get_events_values(
        data=df_arr,
        dates=df_dates,
        arr_dates_oe=arr_dates,
        time_resolution=tres,
    )
    df_all = compute_erosivity(df_all)

    df_erosive = get_only_erosivity_events(
        df_all,
        imax_col=imax_col,
        intensity_threshold=thr_imax15,
        accum_threshold=accum_threshold,
        use_both_thresholds=True,
    )

    mask_i = df_erosive[imax_col] >= thr_imax15
    mask_d = df_erosive['event_depth'] >= accum_threshold

    results_15_i30[label] = {
        'df_all':     df_all,
        'df_erosive': df_erosive,
        'imax_only':  df_erosive[mask_i & ~mask_d],
        'both':       df_erosive[mask_i & mask_d],
        'depth_only': df_erosive[~mask_i & mask_d],
        'none':       df_all[
            ~(df_all[imax_col] >= thr_imax15)
            & ~(df_all['event_depth'] >= accum_threshold)
        ],
        'imax_col':   imax_col,
    }

print("\nTotal event counts — IMax15@imax_30 (all years combined):")
for label, r in results_15_i30.items():
    n_all = len(r['df_all'])
    n_erosive = len(r['df_erosive'])
    n1 = len(r['imax_only'])
    n12 = len(r['both'])
    n2 = len(r['depth_only'])
    print(
        f"  {label:15s} all={n_all:5d}  erosive={n_erosive:5d}"
        f"  imax_only={n1:4d}  both={n12:4d}  depth_only={n2:4d}"
    )


# Figure 6: stacked horizontal bar — IMax15@imax_30
labels_bar_i30 = list(results_15_i30.keys())
n1_i30  = [len(results_15_i30[l]['imax_only'])  for l in labels_bar_i30]
n12_i30 = [len(results_15_i30[l]['both'])       for l in labels_bar_i30]
n2_i30  = [len(results_15_i30[l]['depth_only']) for l in labels_bar_i30]
totals_i30 = [
    a + b + c for a, b, c in zip(n1_i30, n12_i30, n2_i30)
]

y_i30 = range(len(labels_bar_i30))
fig6, ax6 = plt.subplots(figsize=(10, 6))
fig6.suptitle(
    f"{station_num} — Erosive event counts by zone  "
    f"(IMax30 ≥ {thr_imax15} mm/h, old approach)",
    fontsize=13, fontweight='bold',
)

ax6.barh(y_i30, n1_i30, color=COLORS['imax_only'], label='IMax only')
ax6.barh(
    y_i30, n12_i30, color=COLORS['both'],
    left=n1_i30, label='Both',
)
ax6.barh(
    y_i30, n2_i30, color=COLORS['depth_only'],
    left=[a + b for a, b in zip(n1_i30, n12_i30)],
    label='Depth only',
)

for i, (v1, v12, v2, tot) in enumerate(
    zip(n1_i30, n12_i30, n2_i30, totals_i30)
):
    for val, left in [
        (v1,  0),
        (v12, v1),
        (v2,  v1 + v12),
    ]:
        if val == 0:
            continue
        pct = val / tot * 100
        cx = left + val / 2
        ax6.text(
            cx, i, f'{val}\n({pct:.0f}%)',
            ha='center', va='center',
            fontsize=7, color='white', fontweight='bold',
        )

ax6.invert_yaxis()
ax6.set_yticks(list(y_i30))
ax6.set_yticklabels(labels_bar_i30)
ax6.set_xlabel('Total event count (all years)')
ax6.set_xlim(0, 1000)
ax6.legend(
    loc='upper center', bbox_to_anchor=(0.5, -0.12),
    ncol=3, fontsize=9,
)
ax6.grid(axis='x', alpha=0.4)

plt.tight_layout()
if save_results:
    fig6.savefig(
        os.path.join(_FIG, '02_fig3.jpg'),
        format='jpeg', dpi=300,
    )
plt.show()

# %%
# == # == # == # == # == # == # == # == # == # ==
# == # COMBINED SCATTER: 5 rows × 3 columns # ==
# == # == # == # == # == # == # == # == # == # ==

_APPROACHES = [
    {
        'col_title':  f'IMax30 ≥ {thr_imax30} mm/h',
        'results':    results,
        'thr_i':      thr_imax30,
        'hide':       set(),
        'thr_color':  COLORS['imax_only'],
        'thr_lw':     0.8,
    },
    {
        'col_title':  f'IMax15 ≥ {thr_imax15} mm/h',
        'results':    results_15,
        'thr_i':      thr_imax15,
        'hide':       {'30 min', '60 min'},
        'thr_color':  'green',
        'thr_lw':     2.0,
    },
    {
        'col_title':  f'IMax30 ≥ {thr_imax15} mm/h',
        'results':    results_15_i30,
        'thr_i':      thr_imax15,
        'hide':       set(),
        'thr_color':  COLORS['imax_only'],
        'thr_lw':     0.8,
    },
]
_ROW_LABELS = ['5 min', '15 min', '30 min', '60 min']

fig_all, axs_all = plt.subplots(4, 3, figsize=(19.2, 10.8))

for col_idx, appr in enumerate(_APPROACHES):
    for row_idx, res_label in enumerate(_ROW_LABELS):
        ax = axs_all[row_idx, col_idx]
        if res_label in appr['hide']:
            ax.set_visible(False)
            continue
        r = appr['results'].get(res_label)
        if r is None:
            ax.set_visible(False)
            continue

        ic = r['imax_col']
        n1  = len(r['imax_only'])
        n12 = len(r['both'])
        n2  = len(r['depth_only'])
        r_annual = (
            r['df_erosive']
            .groupby(r['df_erosive']['event_start'].dt.year)['erosivity_US']
            .sum()
            .reindex(all_years, fill_value=0)
            .mean()
        )

        ax.scatter(
            r['none']['event_depth'], r['none'][ic],
            color=COLORS['none'], s=10, label='Non-erosive', zorder=1,
        )
        ax.scatter(
            r['imax_only']['event_depth'], r['imax_only'][ic],
            color=COLORS['imax_only'], s=13,
            label=f'IMax only ({n1})', zorder=2,
        )
        ax.scatter(
            r['both']['event_depth'], r['both'][ic],
            color=COLORS['both'], s=13, marker='s',
            label=f'Both ({n12})', zorder=3,
        )
        ax.scatter(
            r['depth_only']['event_depth'], r['depth_only'][ic],
            color=COLORS['depth_only'], s=15, marker='^',
            label=f'Depth only ({n2})', zorder=4,
        )
        ax.axhline(
            r.get('thr_imax30', appr['thr_i']),
            color=COLORS['imax_only'], linestyle='--', linewidth=0.8,
        )
        ax.axvline(
            accum_threshold,
            color=COLORS['depth_only'], linestyle='--', linewidth=0.8,
        )
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Event depth [mm]', fontsize=8)
        ax.set_ylabel(f'{ic} [mm/h]', fontsize=8)
        ax.set_title(res_label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(
            fontsize=7, loc='upper left',
            title=f'Erosive: {n1 + n12 + n2}',
            title_fontsize=7,
        )
        ax.text(
            0.98, 0.98, f'R = {r_annual:.0f} MJ·mm/ha/h/yr',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, pad=4, linewidth=0),
        )
        ax.grid(True, alpha=0.4)
        if col_idx == 1:
            for spine in ax.spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(2.5)

plt.tight_layout()
plt.subplots_adjust(top=0.94)
for col_idx, appr in enumerate(_APPROACHES):
    pos = axs_all[0, col_idx].get_position()
    fig_all.text(
        pos.x0 + pos.width / 2, 0.965,
        appr['col_title'],
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
    )
if save_results:
    fig_all.savefig(
        os.path.join(_FIG, '02_fig4.jpg'),
        format='jpeg', dpi=200,
    )
plt.show()

