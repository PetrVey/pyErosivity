# -*- coding: utf-8 -*-
"""
Bootstrap uncertainty analysis for annual erosivity: OBS vs CPM.

Station VE_0091. Two datasets are compared:
  OBS -- 1-hour gauge observations, 1990-2020 (31 years).
  CPM -- ETH CPM Historical simulation, 1996-2005 (10 years).

The intensity threshold and scaling factor are calibrated from the
5-min OBS reference (IMax15 >= 25.4 mm/h, standard RUSLE criterion ii)
and applied identically to both OBS and CPM 60-min data. This ensures
a consistent, bias-corrected comparison between the two datasets.

Bootstrap procedure: calendar years are resampled with replacement
(1000 iterations). The CPM bootstrap uses a pre-defined sample
sequence (randy.txt) shared across ensemble members for a fair
comparison.

Reference: Dallan et al. (2023) https://doi.org/10.5194/hess-27-1133-2023
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyErosivity import (
    remove_incomplete_years,
    get_events,
    get_events_values,
    compute_erosivity,
    get_only_erosivity_events,
    find_optimal_thr_imax30,
    compute_sf_annual_r,
    get_mean_annual_stats,
    bootstrapping_erosivity_60min,
    bootstrapping_erosivity_CPM_60min,
)

station_num = "VE_0091"
save_results = True

# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTINGS # == # == # SETTINGS # == #
separation = 6          # Min dry-spell between events [h]
min_rain = 0.1          # Drizzle threshold [mm]

tres_5min = 5.0         # 5-min time step [min]
tres_60min = 60.0       # 60-min time step [min]

name_col_5min = "vals"  # 5-min OBS precip column
name_col_obs = "vals"   # 1h OBS precip column
name_col_cpm = "pr_new" # CPM precip column

thr_imax15 = 25.4       # Standard IMax15 reference threshold [mm/h]
accum_threshold = 12.7  # Depth criterion [mm]
use_both = True         # Dual criterion: intensity OR depth
# == # == # SETTINGS # == # == # SETTINGS # == #
# == # == # == # == # == # == # == # == # == # ==

COLORS = {
    'none': 'lightgrey',
    'imax_only': 'steelblue',
    'both': 'mediumpurple',
    'depth_only': 'tomato',
}


def _classify(df_all, imax_col, intensity_thr, accum_thr):
    mask_i = df_all[imax_col] >= intensity_thr
    mask_d = df_all['event_depth'] >= accum_thr
    return {
        'imax_only':  df_all[mask_i & ~mask_d],
        'both':       df_all[mask_i & mask_d],
        'depth_only': df_all[~mask_i & mask_d],
        'none':       df_all[~mask_i & ~mask_d],
    }


# %% === STAGE 1: Load 5-min OBS reference (IMax15 >= 25.4 mm/h) ===

data5 = pd.read_parquet(
    f"res/{station_num}_5min_newflag.parguqet.gzip"
)
data5['time'] = pd.to_datetime(data5['time'])
data5 = data5.set_index('time')
data5.loc[data5['flag'] > 0, name_col_5min] = np.nan
data5.loc[data5[name_col_5min] < min_rain, name_col_5min] = 0
data5, _ = remove_incomplete_years(
    data5, name_col_5min, nan_to_zero=True, tolerance=0.1
)
all_years_ref = sorted(data5.index.year.unique())
precip_ref = (
    data5[name_col_5min].groupby(data5.index.year).sum().mean()
)
print(f"5-min ref precip | mean annual total: {precip_ref:.1f} mm")

arr5 = np.array(data5[name_col_5min])
dates5 = np.array(data5.index)

events5 = get_events(
    data=arr5, dates=dates5,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)
df_all5 = get_events_values(
    data=arr5, dates=dates5,
    arr_dates_oe=events5, time_resolution=tres_5min,
)
df_all5 = compute_erosivity(df_all5)

df_ref = get_only_erosivity_events(
    df_all5,
    imax_col='imax_15',
    intensity_threshold=thr_imax15,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)
stats_ref = get_mean_annual_stats(df_ref, all_years=all_years_ref)
ref_mean_annual = stats_ref['n_events']['mean']

print(
    f"5-min reference | ev/yr: {ref_mean_annual:.2f} "
    f"| R: {stats_ref['erosivity']['mean']:.1f} MJ*mm/ha/h/yr"
)


# %% === STAGE 2: Load and process OBS 1h data ===

data_obs = pd.read_parquet(
    f"res/{station_num}_1h_flag.parguqet.gzip"
)
data_obs['time'] = pd.to_datetime(data_obs['time'])
data_obs = data_obs.set_index('time')
data_obs.loc[data_obs['flag'] > 0, name_col_obs] = np.nan
data_obs.loc[data_obs[name_col_obs] < min_rain, name_col_obs] = 0
data_obs, _ = remove_incomplete_years(
    data_obs, name_col_obs, nan_to_zero=True, tolerance=0.1
)
all_years_obs = sorted(data_obs.index.year.unique())
precip_obs = (
    data_obs[name_col_obs].groupby(data_obs.index.year).sum().mean()
)
print(f"OBS 1h precip   | mean annual total: {precip_obs:.1f} mm")

arr_obs = np.array(data_obs[name_col_obs])
dates_obs = np.array(data_obs.index)

events_obs = get_events(
    data=arr_obs, dates=dates_obs,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)
df_all_obs = get_events_values(
    data=arr_obs, dates=dates_obs,
    arr_dates_oe=events_obs, time_resolution=tres_60min,
)
df_all_obs = compute_erosivity(df_all_obs)


# %% === STAGE 3: Calibrate IMax60 threshold from OBS ===

thr_opt, achieved, residual = find_optimal_thr_imax30(
    df_all_obs,
    target_mean_annual=ref_mean_annual,
    imax_col='imax_60',
    use_both_thresholds=use_both,
)
print(
    f"Optimal IMax60  | threshold: {thr_opt:.4f} mm/h "
    f"| achieved: {achieved:.2f} ev/yr | residual: {residual:.2f}"
)


# %% === STAGE 4: Apply threshold, compute SF, build OBS erosivity df ===

df_obs_raw = get_only_erosivity_events(
    df_all_obs,
    imax_col='imax_60',
    intensity_threshold=thr_opt,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)
sf, _, r_obs_annual = compute_sf_annual_r(
    df_ref, df_obs_raw, all_years=all_years_obs
)

df_obs = df_obs_raw.copy()
df_obs['erosivity_US_adj'] = df_obs['erosivity_US'] * sf

stats_obs = get_mean_annual_stats(
    df_obs,
    ei30_col='erosivity_US_adj',
    depth_col='event_depth',
    intensity_col='imax_60',
    all_years=all_years_obs,
)
print(
    f"OBS 1h          | SF = {sf:.4f} "
    f"| R before: {r_obs_annual.mean():.1f} "
    f"| R after: {stats_obs['erosivity']['mean']:.1f}"
    f" MJ*mm/ha/h/yr"
)


# %% === STAGE 5: Load CPM 1h data and apply same threshold + SF ===

data_cpm = pd.read_csv(f"res/{station_num}_ETH_hist.csv")
data_cpm['time'] = pd.to_datetime(data_cpm['time'])
data_cpm = data_cpm.set_index('time')

# --- raw data diagnostic ---
_raw = data_cpm[name_col_cpm]
_nonzero = _raw[_raw > 0]
print(
    f"\nCPM raw ({name_col_cpm}) | "
    f"rows: {len(_raw):,} | "
    f"non-zero: {len(_nonzero):,} ({100*len(_nonzero)/len(_raw):.1f} %)"
)
print(
    f"  min={_raw.min():.4f}  "
    f"p50={_nonzero.quantile(0.50):.4f}  "
    f"p95={_nonzero.quantile(0.95):.4f}  "
    f"p99={_nonzero.quantile(0.99):.4f}  "
    f"max={_raw.max():.4f}"
)
print(f"  time range: {_raw.index[0]} to {_raw.index[-1]}")
print(f"  columns in file: {list(data_cpm.columns)}\n")
# ---------------------------

data_cpm.loc[data_cpm[name_col_cpm] < min_rain, name_col_cpm] = 0
data_cpm, _ = remove_incomplete_years(
    data_cpm, name_col_cpm, nan_to_zero=True, tolerance=0.1
)
all_years_cpm = sorted(data_cpm.index.year.unique())
precip_cpm = (
    data_cpm[name_col_cpm].groupby(data_cpm.index.year).sum().mean()
)
wet_bias = (precip_cpm / precip_obs - 1) * 100
print(
    f"CPM 1h precip   | mean annual total: {precip_cpm:.1f} mm "
    f"| wet bias vs OBS: {wet_bias:+.1f} %"
)

randy = np.loadtxt('res/randy.txt', delimiter=',')
randy = randy.T.astype(np.int32)

arr_cpm = np.array(data_cpm[name_col_cpm])
dates_cpm = np.array(data_cpm.index)

events_cpm = get_events(
    data=arr_cpm, dates=dates_cpm,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)
df_all_cpm = get_events_values(
    data=arr_cpm, dates=dates_cpm,
    arr_dates_oe=events_cpm, time_resolution=tres_60min,
)
df_all_cpm = compute_erosivity(df_all_cpm)

df_cpm_raw = get_only_erosivity_events(
    df_all_cpm,
    imax_col='imax_60',
    intensity_threshold=thr_opt,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)
df_cpm = df_cpm_raw.copy()
df_cpm['erosivity_US_adj'] = df_cpm['erosivity_US'] * sf

stats_cpm = get_mean_annual_stats(
    df_cpm,
    ei30_col='erosivity_US_adj',
    depth_col='event_depth',
    intensity_col='imax_60',
    all_years=all_years_cpm,
)
print(
    f"CPM OBS-cal     | ev/yr: {stats_cpm['n_events']['mean']:.2f} "
    f"| R (adj): {stats_cpm['erosivity']['mean']:.1f}"
    f" MJ*mm/ha/h/yr"
)


# %% === STAGE 5b: CPM-specific calibration (own threshold + own SF) ===

thr_opt_cpm, achieved_cpm, residual_cpm = find_optimal_thr_imax30(
    df_all_cpm,
    target_mean_annual=ref_mean_annual,
    imax_col='imax_60',
    use_both_thresholds=use_both,
)
print(
    f"CPM own IMax60  | threshold: {thr_opt_cpm:.4f} mm/h "
    f"| achieved: {achieved_cpm:.2f} ev/yr "
    f"| residual: {residual_cpm:.2f}"
)

df_cpm_own_raw = get_only_erosivity_events(
    df_all_cpm,
    imax_col='imax_60',
    intensity_threshold=thr_opt_cpm,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)

# SF computed against 5-min OBS reference over the CPM year range
sf_cpm, _, r_cpm_own_annual = compute_sf_annual_r(
    df_ref, df_cpm_own_raw, all_years=all_years_cpm
)

df_cpm_own = df_cpm_own_raw.copy()
df_cpm_own['erosivity_US_adj'] = df_cpm_own['erosivity_US'] * sf_cpm

stats_cpm_own = get_mean_annual_stats(
    df_cpm_own,
    ei30_col='erosivity_US_adj',
    depth_col='event_depth',
    intensity_col='imax_60',
    all_years=all_years_cpm,
)
print(
    f"CPM CPM-cal     | SF = {sf_cpm:.4f} "
    f"| R before: {r_cpm_own_annual.mean():.1f} "
    f"| R after: {stats_cpm_own['erosivity']['mean']:.1f}"
    f" MJ*mm/ha/h/yr"
)


# %% === STAGE 6: Classification scatter OBS vs CPM ===

scatter_panels = [
    {
        'title': f'OBS 1h | IMax60 >= {thr_opt:.2f} mm/h',
        'df_all': df_all_obs,
        'thr': thr_opt,
        'stats': stats_obs,
    },
    {
        'title': (
            f'CPM 1h | OBS-cal\nIMax60 >= {thr_opt:.2f} mm/h'
        ),
        'df_all': df_all_cpm,
        'thr': thr_opt,
        'stats': stats_cpm,
    },
    {
        'title': (
            f'CPM 1h | CPM-cal\nIMax60 >= {thr_opt_cpm:.2f} mm/h'
        ),
        'df_all': df_all_cpm,
        'thr': thr_opt_cpm,
        'stats': stats_cpm_own,
    },
]

fig_sc, axes_sc = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)

for ax, p in zip(axes_sc, scatter_panels):
    z = _classify(
        p['df_all'], 'imax_60', p['thr'], accum_threshold
    )
    n_im = len(z['imax_only'])
    n_bo = len(z['both'])
    n_de = len(z['depth_only'])

    ax.scatter(
        z['none']['event_depth'], z['none']['imax_60'],
        color=COLORS['none'], s=6, alpha=0.3,
        label='Non-erosive', zorder=1,
    )
    ax.scatter(
        z['imax_only']['event_depth'], z['imax_only']['imax_60'],
        color=COLORS['imax_only'], s=18, marker='o',
        label=f'IMax only ({n_im})', zorder=3,
    )
    ax.scatter(
        z['both']['event_depth'], z['both']['imax_60'],
        color=COLORS['both'], s=18, marker='s',
        label=f'Both ({n_bo})', zorder=3,
    )
    ax.scatter(
        z['depth_only']['event_depth'], z['depth_only']['imax_60'],
        color=COLORS['depth_only'], s=18, marker='^',
        label=f'Depth only ({n_de})', zorder=3,
    )
    ax.axhline(
        p['thr'],
        color=COLORS['imax_only'], linestyle='--', linewidth=0.9,
    )
    ax.axvline(
        accum_threshold,
        color=COLORS['depth_only'], linestyle='--', linewidth=0.9,
    )
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Event depth [mm]')
    ax.set_ylabel('imax_60 [mm/h]', fontweight='bold')
    ax.set_title(p['title'], fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.legend(
        fontsize=8, loc='upper left',
        title=f'Erosive: {n_im + n_bo + n_de}',
        title_fontsize=8,
    )
    ax.text(
        0.98, 0.98,
        f'R = {p["stats"]["erosivity"]["mean"]:.0f}'
        f' MJ*mm/ha/h/yr\n'
        f'{p["stats"]["n_events"]["mean"]:.1f} ev/yr',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.75, pad=2, linewidth=0),
    )

plt.suptitle(
    f'{station_num}: Event classification scatter\n'
    f'OBS thr = {thr_opt:.2f} mm/h (SF={sf:.3f}) | '
    f'CPM-cal thr = {thr_opt_cpm:.2f} mm/h (SF={sf_cpm:.3f})',
    fontsize=11, y=1.01,
)
plt.tight_layout()
if save_results:
    fig_sc.savefig('fig/04_fig0.jpg', dpi=150, bbox_inches='tight')
plt.show()


# %% === STAGE 7: Print population statistics ===

for label, stats in [
    ('OBS', stats_obs),
    ('CPM OBS-cal', stats_cpm),
    ('CPM CPM-cal', stats_cpm_own),
]:
    print(f"\n{label} annual statistics:")
    print(
        f"  Events/yr : {stats['n_events']['mean']:.2f}"
        f" +/- {stats['n_events']['std']:.2f}"
    )
    print(
        f"  IMax60    : {stats['intensity']['mean']:.2f}"
        f" +/- {stats['intensity']['std']:.2f} mm/h"
    )
    print(
        f"  Depth     : {stats['depth']['mean']:.1f}"
        f" +/- {stats['depth']['std']:.1f} mm"
    )
    print(
        f"  R-factor  : {stats['erosivity']['mean']:.1f}"
        f" +/- {stats['erosivity']['std']:.1f} MJ*mm/ha/h/yr"
    )


# %% === STAGE 8: Bootstrap OBS ===

df_bs_obs = bootstrapping_erosivity_60min(
    df_obs,
    imax_col='imax_60',
    erosivity_col='erosivity_US_adj',
)


# %% === STAGE 9: Bootstrap CPM (OBS-calibrated) ===

df_bs_cpm = bootstrapping_erosivity_CPM_60min(
    df_cpm,
    randy=randy,
    imax_col='imax_60',
    erosivity_col='erosivity_US_adj',
)


# %% === STAGE 9b: Bootstrap CPM (CPM-calibrated) ===

df_bs_cpm_own = bootstrapping_erosivity_CPM_60min(
    df_cpm_own,
    randy=randy,
    imax_col='imax_60',
    erosivity_col='erosivity_US_adj',
)


# %% === STAGE 10: Plot merged bootstrap distributions ===

# Bootstrap column -> (stats key, axis label)
PLOT_VARS = [
    ('mean_annual_events',       'n_events',
     'Events per year'),
    ('mean_annual_Imax',         'intensity',
     'Mean IMax60 [mm/h]'),
    ('mean_rain_depth',          'depth',
     'Mean event depth [mm]'),
    ('average_annual_erosivity', 'erosivity',
     'Mean annual R [MJ*mm/ha/h/yr]'),
]

obs_label = f'OBS\n({all_years_obs[0]}-{all_years_obs[-1]})'
cpm_obs_label = f'CPM OBS-cal\n({all_years_cpm[0]}-{all_years_cpm[-1]})'
cpm_own_label = f'CPM CPM-cal\n({all_years_cpm[0]}-{all_years_cpm[-1]})'

BOX_COLORS = ['steelblue', 'tomato', 'mediumpurple']
BOX_MEANS = [
    (1, stats_obs,      'steelblue', 'OBS mean'),
    (2, stats_cpm,      'tomato',    'CPM OBS-cal mean'),
    (3, stats_cpm_own,  'mediumpurple', 'CPM CPM-cal mean'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (bs_col, stat_key, ylabel) in enumerate(PLOT_VARS):
    ax = axes[i]

    bp = ax.boxplot(
        [
            df_bs_obs[bs_col].values,
            df_bs_cpm[bs_col].values,
            df_bs_cpm_own[bs_col].values,
        ],
        positions=[1, 2, 3],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
    )
    for box, color in zip(bp['boxes'], BOX_COLORS):
        box.set_facecolor(color)
        box.set_alpha(0.6)

    for pos, stats, color, label in BOX_MEANS:
        ax.scatter(
            pos, stats[stat_key]['mean'],
            color=color, s=80, marker='D', zorder=5,
            label=label,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([obs_label, cpm_obs_label, cpm_own_label])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, linestyle='--', alpha=0.4)
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle(
    f'{station_num}: Bootstrap uncertainty — OBS vs CPM\n'
    f'OBS-cal: thr={thr_opt:.2f} mm/h SF={sf:.3f} | '
    f'CPM-cal: thr={thr_opt_cpm:.2f} mm/h SF={sf_cpm:.3f}',
    fontsize=12,
)
plt.tight_layout()
if save_results:
    fig.savefig('fig/04_fig1.jpg', dpi=150, bbox_inches='tight')
plt.show()
