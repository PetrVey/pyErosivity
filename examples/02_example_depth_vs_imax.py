# -*- coding: utf-8 -*-
"""
Compares IMax30-only vs dual-threshold (IMax30 OR depth) erosivity
event selection across four temporal resolutions (5, 15, 30, 60 min).

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
from pyErosivity import remove_short
from pyErosivity import get_events_values
from pyErosivity import get_only_erosivity_events
from pyErosivity import find_optimal_thr_imax30
from pyErosivity import get_mean_annual_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, '..', 'res')
_OUT = os.path.join(_HERE, '..', 'out')
_FIG = os.path.join(_HERE, '..', 'fig')

#%%
save_results = False
station_num = "VE_0091"
slice_year_from = "1990"
slice_year_to = "2020"

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
separation = 6          # Dry spell between events [hours]
                        # (Wischmeier and Smith, 1958, 1978)
min_rain = 0.1          # Minimum rain depth [mm]
min_ev_dur = 30         # Minimum event duration [min]
name_col = "vals"       # Column with precipitation data
thr_imax30 = 12.7       # Wischmeier IMax30 threshold [mm/h]
accum_threshold = 12.7  # Min accumulated event depth [mm]
                        # (Wischmeier & Smith 1978)

# Resolutions to analyse — resample rule, time step, accumulation window
# For 60-min: duration=60 because IMax30=IMax60 at hourly resolution
RESOLUTIONS = [
    {'label': '5 min',  'rule': None,    'tres': 5.0,  'dur': 30},
    {'label': '10 min', 'rule': '10min', 'tres': 10.0, 'dur': 30},
    {'label': '15 min', 'rule': '15min', 'tres': 15.0, 'dur': 30},
    {'label': '30 min', 'rule': '30min', 'tres': 30.0, 'dur': 30},
    {'label': '60 min', 'rule': '60min', 'tres': 60.0, 'dur': 60},
]
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

#%%
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

#%%
# Loop over resolutions, compute Venn zones for each
results = {}
table_rows = []
start_time = time.time()

all_years = sorted(
    pd.date_range(slice_year_from, slice_year_to, freq='YS').year
)

for res in RESOLUTIONS:
    label = res['label']
    tres = res['tres']
    dur = res['dur']

    if res['rule'] is None:
        df_res = data_base[[name_col]].copy()
    else:
        df_res = (
            data_base[[name_col]]
            .resample(res['rule'])
            .sum()
        )

    df_arr = np.array(df_res[name_col])
    df_dates = np.array(df_res.index)

    idx_events = get_events(
        data=df_arr,
        dates=df_dates,
        separation=separation,
        min_rain=min_rain,
        name_col=name_col,
        check_gaps=False,
    )

    _, arr_dates, _ = remove_short(
        idx_events,
        time_resolution=tres,
        min_ev_dur=min_ev_dur,
    )

    dict_events = get_events_values(
        data=df_arr,
        dates=df_dates,
        arr_dates_oe=arr_dates,
        durations=[dur],
        time_resolution=tres,
    )

    df_all = dict_events[str(dur)].copy()

    df_erosive = get_only_erosivity_events(
        df_all,
        use_both_thresholds=True,
        intensity_threshold=thr_imax30,
        accum_threshold=accum_threshold,
    )

    mask_i = df_erosive['intensity_per_hour'] >= thr_imax30
    mask_d = df_erosive['event_depth'] >= accum_threshold

    results[label] = {
        'df_all':      df_all,
        'df_erosive':  df_erosive,
        'imax_only':   df_erosive[mask_i & ~mask_d],
        'both':        df_erosive[mask_i & mask_d],
        'depth_only':  df_erosive[~mask_i & mask_d],
        'none':        df_all[
            ~(df_all['intensity_per_hour'] >= thr_imax30)
            & ~(df_all['event_depth'] >= accum_threshold)
        ],
    }

    def mean_annual(df):
        return (
            df.groupby(df['event_start'].dt.year)
            .size()
            .reindex(all_years, fill_value=0)
            .mean()
        )

    m0   = mean_annual(results[label]['none'])
    m1   = mean_annual(results[label]['imax_only'])
    m12  = mean_annual(results[label]['both'])
    m2   = mean_annual(results[label]['depth_only'])
    mtot = m0 + m1 + m12 + m2

    r_factor = get_mean_annual_stats(
        df_erosive, all_years=all_years,
    )['erosivity']['mean']

    m_erosive = m1 + m12 + m2
    table_rows.append({
        'Resolution':   label,
        'Total':        round(mtot, 1),
        'Non-erosive':  round(m0, 1),
        'Erosive':      round(m_erosive, 1),
        'IMax30 only':  round(m1, 1),
        'Both':         round(m12, 1),
        'Depth only':   round(m2, 1),
        'R-factor':     round(r_factor, 1),
    })

# Optimise IMax30 threshold for 60-min data to match 5-min mean annual
# event count
df_5_erosive = results['5 min']['df_erosive']
target_mean_annual = (
    df_5_erosive
    .groupby(df_5_erosive['event_start'].dt.year)
    .size()
    .reindex(all_years, fill_value=0)
    .mean()
)
df_all_60 = results['60 min']['df_all']

thr_opt, achieved, residual = find_optimal_thr_imax30(
    df_all_60,
    target_mean_annual,
    use_both_thresholds=True,
)
print(
    f"60 min opt | target: {target_mean_annual:.2f} ev/yr | "
    f"achieved: {achieved:.2f} ev/yr | thr: {thr_opt:.4f} mm/h | "
    f"residual: {residual:.2f}"
)

df_erosive_opt = get_only_erosivity_events(
    df_all_60,
    use_both_thresholds=True,
    intensity_threshold=thr_opt,
    accum_threshold=accum_threshold,
)

mask_i_opt = df_erosive_opt['intensity_per_hour'] >= thr_opt
mask_d_opt = df_erosive_opt['event_depth'] >= accum_threshold

results['60 min (opt)'] = {
    'df_all':     df_all_60,
    'df_erosive': df_erosive_opt,
    'imax_only':  df_erosive_opt[mask_i_opt & ~mask_d_opt],
    'both':       df_erosive_opt[mask_i_opt & mask_d_opt],
    'depth_only': df_erosive_opt[~mask_i_opt & mask_d_opt],
    'none':       df_all_60[
        ~(df_all_60['intensity_per_hour'] >= thr_opt)
        & ~(df_all_60['event_depth'] >= accum_threshold)
    ],
    'thr_imax30': thr_opt,
}

m0_o  = mean_annual(results['60 min (opt)']['none'])
m1_o  = mean_annual(results['60 min (opt)']['imax_only'])
m12_o = mean_annual(results['60 min (opt)']['both'])
m2_o  = mean_annual(results['60 min (opt)']['depth_only'])
r_factor_opt = get_mean_annual_stats(
    df_erosive_opt, all_years=all_years,
)['erosivity']['mean']

table_rows.append({
    'Resolution':  '60 min (opt)',
    'Total':       round(m0_o + m1_o + m12_o + m2_o, 1),
    'Non-erosive': round(m0_o, 1),
    'Erosive':     round(m1_o + m12_o + m2_o, 1),
    'IMax30 only': round(m1_o, 1),
    'Both':        round(m12_o, 1),
    'Depth only':  round(m2_o, 1),
    'R-factor':    round(r_factor_opt, 1),
})

print(f"Total calculation time: {time.time() - start_time:.2f} s\n")
df_table = pd.DataFrame(table_rows).set_index('Resolution')
df_table.columns = [
    *df_table.columns[:-1],
    'R-factor\n[MJ mm ha-1 h-1 yr-1]',
]
print(df_table.to_string())

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # PLOTS # == # == # PLOTS # == # == # ==
# == # == # == # == # == # == # == # == # == # ==

COLORS = {
    'none':        'lightgrey',
    'imax_only':   'steelblue',
    'both':        'mediumpurple',
    'depth_only':  'tomato',
}

# Figure 1: scatter per resolution (2x2)
fig1, axs = plt.subplots(2, 3, figsize=(16, 9))
fig1.suptitle(
    f"{station_num} — Event space by resolution",
    fontsize=13, fontweight='bold',
)

for ax, (label, r) in zip(axs.flat, results.items()):
    n1  = len(r['imax_only'])
    n12 = len(r['both'])
    n2  = len(r['depth_only'])

    ax.scatter(
        r['none']['event_depth'],
        r['none']['intensity_per_hour'],
        color=COLORS['none'], s=12, label='Non-erosive', zorder=1,
    )
    ax.scatter(
        r['imax_only']['event_depth'],
        r['imax_only']['intensity_per_hour'],
        color=COLORS['imax_only'], s=15,
        label=f'IMax30 only ({n1})', zorder=2,
    )
    ax.scatter(
        r['both']['event_depth'],
        r['both']['intensity_per_hour'],
        color=COLORS['both'], s=15, marker='s',
        label=f'Both ({n12})', zorder=3,
    )
    ax.scatter(
        r['depth_only']['event_depth'],
        r['depth_only']['intensity_per_hour'],
        color=COLORS['depth_only'], s=18, marker='^',
        label=f'Depth only ({n2})', zorder=4,
    )
    ax.axhline(
        r.get('thr_imax30', thr_imax30),
        color=COLORS['imax_only'],
        linestyle='--', linewidth=0.8,
    )
    ax.axvline(
        accum_threshold, color=COLORS['depth_only'],
        linestyle='--', linewidth=0.8,
    )
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_xlabel('Total event depth, event_depth [mm]')
    ax.set_ylabel('IMax30 [mm/h]')
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.grid(True)

plt.tight_layout()
if save_results:
    fig1.savefig(
        os.path.join(_FIG, 'fig02a_scatter_by_resolution.jpeg'),
        format='jpeg', dpi=300,
    )
plt.show()

# Figure 2: stacked bar — zone composition per resolution
years = sorted(data_base.index.year.unique())
res_labels = [r['label'] for r in RESOLUTIONS]

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle(
    f"{station_num} — Annual erosive events by resolution & criterion",
    fontsize=13, fontweight='bold',
)

n_res = len(RESOLUTIONS)
n_years = len(years)
group_w = 0.8
bar_w = group_w / n_res
x = np.arange(n_years)

for i, (label, r) in enumerate(results.items()):
    offset = (i - n_res / 2 + 0.5) * bar_w

    def annual(df):
        return (
            df.groupby(df['event_start'].dt.year)
            .size()
            .reindex(years, fill_value=0)
            .values
        )

    c1  = annual(r['imax_only'])
    c12 = annual(r['both'])
    c2  = annual(r['depth_only'])

    ax2.bar(
        x + offset, c1, width=bar_w,
        color=COLORS['imax_only'],
        label='IMax30 only' if i == 0 else '_',
    )
    ax2.bar(
        x + offset, c12, width=bar_w, bottom=c1,
        color=COLORS['both'],
        label='Both' if i == 0 else '_',
    )
    ax2.bar(
        x + offset, c2, width=bar_w, bottom=c1 + c12,
        color=COLORS['depth_only'],
        label='Depth only' if i == 0 else '_',
    )
    for xi, yi in zip(x + offset, c1 + c12 + c2):
        ax2.text(
            xi, yi + 0.2, label, ha='center',
            va='bottom', fontsize=6, rotation=90,
        )

ax2.set_xticks(x)
ax2.set_xticklabels(years, rotation=45)
ax2.set_xlabel('Year')
ax2.set_ylabel('Number of erosive events')
ax2.legend(fontsize=9)
ax2.grid(True, axis='y')

plt.tight_layout()
if save_results:
    fig2.savefig(
        os.path.join(_FIG, 'fig02b_bar_by_resolution.jpeg'),
        format='jpeg', dpi=300,
    )
plt.show()
