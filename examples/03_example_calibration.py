# -*- coding: utf-8 -*-
"""
Calibration example: matching 60-min erosivity to a 5-min reference.

Station VE_0091, 1990-2020 (31 years), 5-min gauge data.

Steps:
  1. 5-min data, standard IMax15 >= 25.4 mm/h criterion  ->  reference.
  2. Aggregate to 60-min; apply the same pipeline with an unadjusted
     12.7 mm/h threshold.  At hourly resolution IMax60 rarely exceeds
     12.7 mm/h, so the intensity zone empties and R-factor drops sharply.
  3. find_optimal_thr_imax30 sweeps every unique IMax60 value and returns
     the threshold that minimises the gap to the reference event count.

Two scaling factor (SF) approaches are compared:

  SF-R  (annual R-factor) -- pair ref and target year by year, compute
        SF = mean(R_ref_annual) / mean(R_target_annual).
        Works entirely at the R-factor level; no event matching needed.

  SF-EI (per-event EI) -- match events by date, compute
        SF = mean(EI_ref_matched) / mean(EI_target_matched).
        Works inside the event population; captures within-event bias.

Both approaches should give similar SFs; small differences arise because
SF-R averages over years while SF-EI averages over individual events.

Reference: Fischer et al. (2018) HESS 22, 6505-6518.
           https://doi.org/10.5194/hess-22-6505-2018
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
    compute_sf_per_event,
)

station_num = "VE_0091"
save_results = True

# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTINGS # == # == # SETTINGS # == #
separation = 6      # Min dry-spell between events [h]
min_rain = 0.1      # Drizzle threshold [mm]
name_col = "vals"   # Precipitation column name

tres_5min = 5.0     # 5-min time resolution [min]
tres_60min = 60.0   # 60-min time resolution [min]

thr_imax15 = 25.4       # Standard IMax15 threshold [mm/h]
thr_imax60_naive = 12.7  # Naive IMax60 threshold — same value, wider window
accum_threshold = 12.7  # Minimum event depth for depth criterion [mm]
use_both = True         # Dual criterion: intensity OR depth
# == # == # SETTINGS # == # == # SETTINGS # == #
# == # == # == # == # == # == # == # == # == # ==

COLORS = {
    'none': 'lightgrey',
    'imax_only': 'steelblue',
    'both': 'mediumpurple',
    'depth_only': 'tomato',
}


# %% === STAGE 1: Load and clean 5-min data ===

data = pd.read_parquet(f"res/{station_num}_5min_newflag.parguqet.gzip")
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')

data.loc[data['flag'] > 0, name_col] = np.nan
data.loc[data[name_col] < min_rain, name_col] = 0

data, _ = remove_incomplete_years(
    data, name_col, nan_to_zero=True, tolerance=0.1
)
all_years = sorted(data.index.year.unique())
n_years = len(all_years)


# %% === STAGE 2: 5-min reference — IMax15 >= 25.4 mm/h ===

arr_5 = np.array(data[name_col])
dates_5 = np.array(data.index)

events_5 = get_events(
    data=arr_5, dates=dates_5,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)
df_all_5 = get_events_values(
    data=arr_5, dates=dates_5,
    arr_dates_oe=events_5, time_resolution=tres_5min,
)
df_all_5 = compute_erosivity(df_all_5)

df_ref = get_only_erosivity_events(
    df_all_5,
    imax_col='imax_15',
    intensity_threshold=thr_imax15,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)

ref_mean_annual = (
    df_ref.groupby(df_ref['event_start'].dt.year)
    .size().reindex(all_years, fill_value=0).mean()
)
ref_rfactor = df_ref['erosivity_US'].sum() / n_years

print(
    f"5-min reference | events/yr: {ref_mean_annual:.2f} "
    f"| R-factor: {ref_rfactor:.1f} MJ*mm/ha/h/yr"
)


# %% === STAGE 3: Aggregate 5-min to 60-min ===

data_60 = data[[name_col]].resample('60min').sum()

arr_60 = np.array(data_60[name_col])
dates_60 = np.array(data_60.index)

events_60 = get_events(
    data=arr_60, dates=dates_60,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)
df_all_60 = get_events_values(
    data=arr_60, dates=dates_60,
    arr_dates_oe=events_60, time_resolution=tres_60min,
)
df_all_60 = compute_erosivity(df_all_60)


# %% === STAGE 4: 60-min naive — IMax60 >= 12.7 mm/h (unadjusted) ===

df_naive = get_only_erosivity_events(
    df_all_60,
    imax_col='imax_60',
    intensity_threshold=thr_imax60_naive,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)

naive_rfactor = df_naive['erosivity_US'].sum() / n_years

print(
    f"60-min naive    | events/yr: "
    f"{df_naive.groupby(df_naive['event_start'].dt.year).size().reindex(all_years, fill_value=0).mean():.2f} "
    f"| R-factor: {naive_rfactor:.1f} MJ*mm/ha/h/yr"
)


# %% === STAGE 5: Calibrate — find optimal IMax60 threshold ===

thr_opt, achieved, residual = find_optimal_thr_imax30(
    df_all_60,
    target_mean_annual=ref_mean_annual,
    imax_col='imax_60',
    use_both_thresholds=use_both,
)
print(
    f"Optimal IMax60  | threshold: {thr_opt:.4f} mm/h "
    f"| achieved: {achieved:.2f} ev/yr | residual: {residual:.2f}"
)

df_opt = get_only_erosivity_events(
    df_all_60,
    imax_col='imax_60',
    intensity_threshold=thr_opt,
    accum_threshold=accum_threshold,
    use_both_thresholds=use_both,
)

opt_rfactor = df_opt['erosivity_US'].sum() / n_years

print(
    f"60-min opt      | events/yr: "
    f"{df_opt.groupby(df_opt['event_start'].dt.year).size().reindex(all_years, fill_value=0).mean():.2f} "
    f"| R-factor: {opt_rfactor:.1f} MJ*mm/ha/h/yr"
)


# %% === STAGE 6: Plot — event classification scatter (fig1) ===

def classify(df_all, imax_col, intensity_thr, accum_thr):
    """Split df_all into imax_only / both / depth_only / none."""
    mask_i = df_all[imax_col] >= intensity_thr
    mask_d = df_all['event_depth'] >= accum_thr
    return {
        'imax_only': df_all[mask_i & ~mask_d],
        'both': df_all[mask_i & mask_d],
        'depth_only': df_all[~mask_i & mask_d],
        'none': df_all[~mask_i & ~mask_d],
    }


zones_ref = classify(df_all_5, 'imax_15', thr_imax15, accum_threshold)
zones_naive = classify(
    df_all_60, 'imax_60', thr_imax60_naive, accum_threshold
)
zones_opt = classify(df_all_60, 'imax_60', thr_opt, accum_threshold)

naive_evyr = (
    df_naive.groupby(df_naive['event_start'].dt.year)
    .size().reindex(all_years, fill_value=0).mean()
)
opt_evyr = (
    df_opt.groupby(df_opt['event_start'].dt.year)
    .size().reindex(all_years, fill_value=0).mean()
)

panels = [
    {
        'title': f'5 min - IMax15 >= {thr_imax15} mm/h (standard)',
        'zones': zones_ref,
        'imax_col': 'imax_15',
        'thr_i': thr_imax15,
        'rfactor': ref_rfactor,
        'evyr': ref_mean_annual,
    },
    {
        'title': f'60 min - IMax60 >= {thr_imax60_naive} mm/h (naive)',
        'zones': zones_naive,
        'imax_col': 'imax_60',
        'thr_i': thr_imax60_naive,
        'rfactor': naive_rfactor,
        'evyr': naive_evyr,
    },
    {
        'title': f'60 min - IMax60 >= {thr_opt:.2f} mm/h (calibrated)',
        'zones': zones_opt,
        'imax_col': 'imax_60',
        'thr_i': thr_opt,
        'rfactor': opt_rfactor,
        'evyr': opt_evyr,
    },
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

for ax, p in zip(axes, panels):
    z = p['zones']
    ic = p['imax_col']
    n_im = len(z['imax_only'])
    n_bo = len(z['both'])
    n_de = len(z['depth_only'])

    ax.scatter(
        z['none']['event_depth'], z['none'][ic],
        color=COLORS['none'], s=6, alpha=0.3,
        label='Non-erosive', zorder=1,
    )
    ax.scatter(
        z['imax_only']['event_depth'], z['imax_only'][ic],
        color=COLORS['imax_only'], s=18, marker='o',
        label=f'IMax only ({n_im})', zorder=3,
    )
    ax.scatter(
        z['both']['event_depth'], z['both'][ic],
        color=COLORS['both'], s=18, marker='s',
        label=f'Both ({n_bo})', zorder=3,
    )
    ax.scatter(
        z['depth_only']['event_depth'], z['depth_only'][ic],
        color=COLORS['depth_only'], s=18, marker='^',
        label=f'Depth only ({n_de})', zorder=3,
    )
    ax.axhline(
        p['thr_i'],
        color=COLORS['imax_only'], linestyle='--', linewidth=0.9,
    )
    ax.axvline(
        accum_threshold,
        color=COLORS['depth_only'], linestyle='--', linewidth=0.9,
    )
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Event depth [mm]')
    ax.set_ylabel(f'{ic} [mm/h]', fontweight='bold')
    ax.set_title(p['title'], fontsize=10)
    ax.grid(True, alpha=0.35)
    ax.legend(
        fontsize=8, loc='upper left',
        title=f'Erosive: {n_im + n_bo + n_de}',
        title_fontsize=8,
    )
    ax.text(
        0.98, 0.98,
        f'R = {p["rfactor"]:.0f} MJ*mm/ha/h/yr\n{p["evyr"]:.1f} ev/yr',
        transform=ax.transAxes, ha='right', va='top', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.75, pad=2, linewidth=0),
    )

plt.suptitle(
    f'{station_num} - Effect of temporal resolution and threshold '
    f'calibration on erosivity event selection',
    fontsize=11, y=1.01,
)
plt.tight_layout()
if save_results:
    fig.savefig('fig/03_fig1.jpg', dpi=150, bbox_inches='tight')
plt.show()


# %% === STAGE 7: Scaling factors — approach 1 (annual R-factor) ===

sf_r_naive, r_ref_annual, r_naive_annual = compute_sf_annual_r(
    df_ref, df_naive, all_years=all_years
)
sf_r_opt, _, r_opt_annual = compute_sf_annual_r(
    df_ref, df_opt, all_years=all_years
)

print(
    f"\nSF-R naive      | {sf_r_naive:.4f} "
    f"| R corrected: {naive_rfactor * sf_r_naive:.1f} MJ*mm/ha/h/yr"
)
print(
    f"SF-R opt        | {sf_r_opt:.4f} "
    f"| R corrected: {opt_rfactor * sf_r_opt:.1f} MJ*mm/ha/h/yr"
)


# %% === STAGE 8: Scaling factors — approach 2 (per-event EI) ===

sf_ei_naive, ei_ref_n, ei_naive, n_naive = compute_sf_per_event(
    df_ref, df_naive
)
sf_ei_opt, ei_ref_o, ei_opt, n_opt = compute_sf_per_event(
    df_ref, df_opt
)

print(
    f"\nSF-EI naive     | {sf_ei_naive:.4f} "
    f"| matched events: {n_naive}"
)
print(
    f"SF-EI opt       | {sf_ei_opt:.4f} "
    f"| matched events: {n_opt}"
)


# %% === STAGE 9: Plot — SF scatter figures (fig2) ===

fig_sf, axs = plt.subplots(2, 2, figsize=(12, 10))

# --- Row 0: annual R-factor scatter ---
titles_r = [
    f'Naive (IMax60 >= {thr_imax60_naive} mm/h)',
    f'Calibrated (IMax60 >= {thr_opt:.2f} mm/h)',
]
datasets_r = [
    (r_ref_annual, r_naive_annual, sf_r_naive),
    (r_ref_annual, r_opt_annual, sf_r_opt),
]

for col, (r_ref_s, r_tgt_s, sf) in enumerate(datasets_r):
    ax = axs[0, col]
    ax.scatter(r_ref_s, r_tgt_s, color='steelblue', s=40, zorder=3)
    lim = max(r_ref_s.max(), r_tgt_s.max()) * 1.08
    xline = np.linspace(0, lim, 100)
    ax.plot(xline, xline, color='grey', linestyle='--',
            linewidth=0.9, label='1:1')
    ax.plot(xline, xline / sf, color='tomato', linewidth=1.5,
            label=f'SF = {sf:.3f}')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Annual R — 5 min ref [MJ*mm/ha/h/yr]')
    ax.set_ylabel('Annual R — 60 min [MJ*mm/ha/h/yr]')
    ax.set_title(f'SF-R | {titles_r[col]}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

# --- Row 1: per-event EI scatter ---
titles_ei = [
    f'Naive (IMax60 >= {thr_imax60_naive} mm/h) | {n_naive} events',
    f'Calibrated (IMax60 >= {thr_opt:.2f} mm/h) | {n_opt} events',
]
datasets_ei = [
    (ei_ref_n, ei_naive, sf_ei_naive),
    (ei_ref_o, ei_opt, sf_ei_opt),
]

for col, (ei_ref_s, ei_tgt_s, sf) in enumerate(datasets_ei):
    ax = axs[1, col]
    ax.scatter(ei_ref_s, ei_tgt_s, color='mediumpurple',
               s=15, alpha=0.6, zorder=3)
    lim = max(ei_ref_s.max(), ei_tgt_s.max()) * 1.08
    xline = np.linspace(0, lim, 100)
    ax.plot(xline, xline, color='grey', linestyle='--',
            linewidth=0.9, label='1:1')
    ax.plot(xline, xline / sf, color='tomato', linewidth=1.5,
            label=f'SF = {sf:.3f}')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('EI — 5 min ref [MJ*mm/ha/h]')
    ax.set_ylabel('EI — 60 min [MJ*mm/ha/h]')
    ax.set_title(f'SF-EI | {titles_ei[col]}', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

plt.suptitle(
    f'{station_num} - Scaling factor approaches: annual R (top) '
    f'vs per-event EI (bottom)',
    fontsize=11,
)
plt.tight_layout()
if save_results:
    fig_sf.savefig('fig/03_fig2.jpg', dpi=150, bbox_inches='tight')
plt.show()


# %% === STAGE 10: Plot — before/after correction (fig3) ===

FONT_TITLE = 15
FONT_LABEL = 13
FONT_TICK = 12
FONT_NOTE = 13

# Mean annual R after SF-R correction (applied year by year)
r_naive_after_sfr = float((r_naive_annual * sf_r_naive).mean())
r_opt_after_sfr = float((r_opt_annual * sf_r_opt).mean())

# Mean annual R after SF-EI correction (applied to all events)
r_naive_after_sfei = float(
    (df_naive['erosivity_US'] * sf_ei_naive)
    .groupby(df_naive['event_start'].dt.year).sum()
    .reindex(all_years, fill_value=0).mean()
)
r_opt_after_sfei = float(
    (df_opt['erosivity_US'] * sf_ei_opt)
    .groupby(df_opt['event_start'].dt.year).sum()
    .reindex(all_years, fill_value=0).mean()
)


def _panel(ax, x, y, title, xlabel, ylabel, note, color):
    lim = max(x.max(), y.max()) * 1.1
    xline = np.linspace(0, lim, 100)
    ax.scatter(x, y, color=color, s=30, alpha=0.7, zorder=3)
    ax.plot(xline, xline, color='grey', linestyle='--',
            linewidth=1.2, label='1:1', zorder=2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_TICK)
    ax.legend(fontsize=FONT_NOTE)
    ax.grid(True, alpha=0.35)
    ax.text(
        0.05, 0.95, note,
        transform=ax.transAxes, fontsize=FONT_NOTE,
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.8, pad=4, linewidth=0),
    )


fig3, axes3 = plt.subplots(2, 4, figsize=(26, 13))
plt.subplots_adjust(
    left=0.05, right=0.99, top=0.88, bottom=0.08,
    wspace=0.52, hspace=0.52,
)

# --- Row 0: SF-R — annual R year by year ---
_panel(
    axes3[0, 0], r_ref_annual, r_naive_annual,
    title=f'BEFORE | Naive\nIMax60 >= {thr_imax60_naive} mm/h',
    xlabel='R ref 5 min [MJ*mm/ha/h/yr]',
    ylabel='R target 60 min [MJ*mm/ha/h/yr]',
    note=f'Mean annual R\n= {naive_rfactor:.0f} MJ*mm/ha/h/yr',
    color='steelblue',
)
_panel(
    axes3[0, 1], r_ref_annual, r_naive_annual * sf_r_naive,
    title=f'AFTER | SF-R = {sf_r_naive:.3f}',
    xlabel='R ref 5 min [MJ*mm/ha/h/yr]',
    ylabel='R corrected 60 min [MJ*mm/ha/h/yr]',
    note=f'Mean annual R\n= {r_naive_after_sfr:.0f} MJ*mm/ha/h/yr',
    color='steelblue',
)
_panel(
    axes3[0, 2], r_ref_annual, r_opt_annual,
    title=f'BEFORE | Calibrated\nIMax60 >= {thr_opt:.2f} mm/h',
    xlabel='R ref 5 min [MJ*mm/ha/h/yr]',
    ylabel='R target 60 min [MJ*mm/ha/h/yr]',
    note=f'Mean annual R\n= {opt_rfactor:.0f} MJ*mm/ha/h/yr',
    color='mediumpurple',
)
_panel(
    axes3[0, 3], r_ref_annual, r_opt_annual * sf_r_opt,
    title=f'AFTER | SF-R = {sf_r_opt:.3f}',
    xlabel='R ref 5 min [MJ*mm/ha/h/yr]',
    ylabel='R corrected 60 min [MJ*mm/ha/h/yr]',
    note=f'Mean annual R\n= {r_opt_after_sfr:.0f} MJ*mm/ha/h/yr',
    color='mediumpurple',
)

# --- Row 1: SF-EI — per-event EI (matched events shown) ---
_panel(
    axes3[1, 0], ei_ref_n, ei_naive,
    title=f'BEFORE | Naive\nIMax60 >= {thr_imax60_naive} mm/h',
    xlabel='EI ref 5 min [MJ*mm/ha/h]',
    ylabel='EI target 60 min [MJ*mm/ha/h]',
    note=f'Mean annual R\n= {naive_rfactor:.0f} MJ*mm/ha/h/yr',
    color='steelblue',
)
_panel(
    axes3[1, 1], ei_ref_n, ei_naive * sf_ei_naive,
    title=f'AFTER | SF-EI = {sf_ei_naive:.3f}',
    xlabel='EI ref 5 min [MJ*mm/ha/h]',
    ylabel='EI corrected 60 min [MJ*mm/ha/h]',
    note=f'Mean annual R\n= {r_naive_after_sfei:.0f} MJ*mm/ha/h/yr',
    color='steelblue',
)
_panel(
    axes3[1, 2], ei_ref_o, ei_opt,
    title=f'BEFORE | Calibrated\nIMax60 >= {thr_opt:.2f} mm/h',
    xlabel='EI ref 5 min [MJ*mm/ha/h]',
    ylabel='EI target 60 min [MJ*mm/ha/h]',
    note=f'Mean annual R\n= {opt_rfactor:.0f} MJ*mm/ha/h/yr',
    color='mediumpurple',
)
_panel(
    axes3[1, 3], ei_ref_o, ei_opt * sf_ei_opt,
    title=f'AFTER | SF-EI = {sf_ei_opt:.3f}',
    xlabel='EI ref 5 min [MJ*mm/ha/h]',
    ylabel='EI corrected 60 min [MJ*mm/ha/h]',
    note=f'Mean annual R\n= {r_opt_after_sfei:.0f} MJ*mm/ha/h/yr',
    color='mediumpurple',
)

# Reference R annotated on every panel for comparison
for ax in axes3.flat:
    ax.text(
        0.05, 0.72,
        f'Ref R = {ref_rfactor:.0f}',
        transform=ax.transAxes, fontsize=FONT_NOTE - 1,
        va='top', color='grey',
    )

# Row labels on left margin
for row, label in enumerate(
    ['SF-R  (annual R-factor)', 'SF-EI  (per-event EI)']
):
    axes3[row, 0].set_ylabel(
        f'{label}\n\n{axes3[row, 0].get_ylabel()}',
        fontsize=FONT_LABEL,
    )

# Draw arrows between before/after panels (cols 0→1 and 2→3)
ax_bg = fig3.add_axes([0, 0, 1, 1], facecolor='none')
ax_bg.set_xlim(0, 1)
ax_bg.set_ylim(0, 1)
ax_bg.axis('off')

for row in range(2):
    for col_l in [0, 2]:
        pos_l = axes3[row, col_l].get_position()
        pos_r = axes3[row, col_l + 1].get_position()
        x_s = pos_l.x1 + 0.004
        x_e = pos_r.x0 - 0.004
        y_m = (pos_l.y0 + pos_l.y1) / 2
        ax_bg.annotate(
            '', xy=(x_e, y_m), xytext=(x_s, y_m),
            arrowprops=dict(
                arrowstyle='->', color='black',
                lw=2.5, mutation_scale=24,
            ),
        )

plt.suptitle(
    f'{station_num} — Before / after scaling factor correction\n'
    f'Top row: SF-R applied to annual R  |  '
    f'Bottom row: SF-EI applied to per-event EI',
    fontsize=16, fontweight='bold',
)
if save_results:
    fig3.savefig('fig/03_fig3.jpg', dpi=150, bbox_inches='tight')
plt.show()
