# -*- coding: utf-8 -*-
"""
Compares pyErosivity output against RIST (Rainfall Intensity
Summarization Tool) for 5-min precipitation data.

INPUT DATA
----------
The RIST input CSV is produced by running 00_export_5min_to_RIST_input.py.
RIST gauge and R-factor settings are documented in fig/:
    RIST_setup_station.jpg         — station / gauge configuration
    RIST_setup_Rfactor_Imax30.jpg  — R-factor settings using IMax30
    RIST_setup_Rfactor_Imax15.jpg  — R-factor settings using IMax15

RIST SETUPS
-----------
RIST was run in two configurations, both applying the same depth
criterion (i):
    (i)  accumulated event depth >= 12.7 mm (0.5 in)
         (Wischmeier & Smith 1958)

They differ in the intensity criterion (ii):
    IMax15 setup: IMax15 >= 25.4 mm/h
        The original USLE criterion — the marginal erosive event
        concentrates 6.35 mm (0.25 in) in 15 min, giving
        IMax15 = 6.35 * 60/15 = 25.4 mm/h
        (Wischmeier & Smith 1978)

    IMax30 setup: IMax30 >= 12.7 mm/h
        Re-expression of the IMax15 criterion at a 30-min window.
        Williams & Sheridan (1991) eq. 7 defines:
            I₃₀ = 2 × max₃₀ΔDₙ
        where max₃₀ΔDₙ is the peak 30-min accumulated depth [mm] and
        the factor 2 = 60/30 is purely a unit conversion
        (depth per 30 min → mm/h).
        Assuming the 6.35 mm burst is the only rain in its 30-min
        window (the other 15 min are dry), the 30-min depth is still
        6.35 mm, giving:
            I₃₀ = 2 × 6.35 mm = 12.7 mm/h
        This is exactly half of IMax15 because the same 6.35 mm is
        averaged over twice the window length:
            IMax15 = 6.35 / (15/60) = 25.4 mm/h
            IMax30 = 6.35 / (30/60) = 12.7 mm/h

This script compares pyErosivity against both RIST setups
(use_both_thresholds=True).

NOTE ON RIST STORM SEPARATION
------------------------------
RIST (and RUSLE) apply a 6-hour separation rule: a storm is split if
any 6-hour period within it accumulates less than 1.27 mm (0.05 in),
and a storm is split when a gap of >= 6 hours (exact boundary included)
contains less than 1.27 mm. RIST performs its calculations internally
in inches with unknown precision, which causes small rounding
differences vs pyErosivity (which works in mm throughout). 
Consequences:
- Per-event depths reported by RIST are slightly lower than pyErosivity
  for the same storm (rounding of mm→inches→mm).
- Borderline events at exactly the erosivity threshold may be excluded
  by RIST but retained by pyErosivity, e.g. depth=12.8 mm rounds to
  0.50 in=12.7 mm in RIST (at threshold, excluded), and imax30=12.8
  mm/h maps to a 30-min depth of 6.4 mm=0.252 in→0.25 in=12.7 mm/h
  (at threshold, excluded). These discrepancies are expected and
  inherent to RIST's inch-based arithmetic.
"""

import os
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

#%%
save_results = False
station_num = "VE_0091"
slice_year_from = "1990"
slice_year_to = "2020"


def read_rist_output(path):
    """
    Parse a RIST 3.99 storm output text file into a DataFrame.

    Locates the event table by finding the first dashed separator line
    (marking the start of data) and the second one (marking the end).
    Everything after the second separator is summary tables which are
    ignored.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    sep_indices = [
        i for i, l in enumerate(lines)
        if l.strip().startswith('---')
    ]
    header_line = lines[sep_indices[0] - 2].split()
    data_lines = lines[sep_indices[0] + 1: sep_indices[1]]
    data_lines = [l for l in data_lines if l.strip()]
    df = pd.DataFrame(
        [l.split() for l in data_lines], columns=header_line
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date")
    df = df.astype(float)
    return df


df_rist_imax30 = read_rist_output(os.path.join(
    _RES,
    f"RIST_{station_num}_Erosive_Events_5minutes_imax30.txt",
))
df_rist_imax30 = df_rist_imax30[slice_year_from:slice_year_to]

df_rist_imax15 = read_rist_output(os.path.join(
    _RES,
    f"RIST_{station_num}_Erosive_Events_5minutes_imax15.txt",
))
df_rist_imax15 = df_rist_imax15[slice_year_from:slice_year_to]

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
separation = 6           # Dry spell between events [hours]
                         # (Wischmeier and Smith, 1958, 1978)
min_rain = 0.1           # Minimum rain depth [mm]
time_resolution = 5.0    # Dataset time resolution [min]
name_col = "vals"        # Column with precipitation data
use_both_thresholds = True   # Depth OR intensity, matching RIST setup
thr_imax30 = 12.7        # Wischmeier IMax30 threshold [mm/h]
imax_col = 'imax_30'     # Intensity column for IMax30 criterion
thr_imax15 = 25.4        # Wischmeier IMax15 threshold [mm/h]
imax_col_15 = 'imax_15'  # Intensity column for IMax15 criterion
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

data = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_5min_newflag.parguqet.gzip")
)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.loc[slice_year_from:slice_year_to]


data.loc[data['flag'] > 0, name_col] = np.nan
data.loc[data[name_col] < min_rain, name_col] = 0

data, time_resolution_check = remove_incomplete_years(
    data, name_col, nan_to_zero=True, tolerance=0.1
)

if time_resolution_check != time_resolution:
    print("There might be inconsistency in dataset time resolution")

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

# Standard 6h separation pipeline
arr_dates = get_events(
    data=df_arr,
    dates=df_dates,
    separation=separation,
    min_rain=min_rain,
    name_col=name_col,
    check_gaps=False,
    time_resolution=time_resolution,
)

df_events = get_events_values(
    data=df_arr,
    dates=df_dates,
    arr_dates_oe=arr_dates,
    time_resolution=time_resolution,
)
df_events = compute_erosivity(df_events, imax_col=imax_col)
df_erosivity_5 = get_only_erosivity_events(
    df_events,
    imax_col=imax_col,
    intensity_threshold=thr_imax30,
    use_both_thresholds=use_both_thresholds,
)

# IMax15 — EI30 is always E*I30 by definition; imax15 is the selection
# criterion only. df_events already has erosivity_US from imax30.
df_erosivity_15 = get_only_erosivity_events(
    df_events,
    imax_col=imax_col_15,
    intensity_threshold=thr_imax15,
    use_both_thresholds=use_both_thresholds,
)



#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # COMPARISON # == # COMPARISON # == # ==
# == # == # == # == # == # == # == # == # == # ==

for df_r in (df_rist_imax30, df_rist_imax15):
    df_r['date'] = df_r.index.date
df_rist_imax30 = df_rist_imax30.rename(columns={"EI30": "RIST_EI30"})
df_rist_imax15 = df_rist_imax15.rename(columns={"EI30": "RIST_EI30"})


def _event_date(ts_series):
    """Use previous day when event starts exactly at midnight."""
    dates = ts_series.dt.date
    midnight = (ts_series.dt.hour == 0) & (ts_series.dt.minute == 0)
    dates[midnight] = (
        ts_series[midnight] - pd.Timedelta(days=1)
    ).dt.date
    return dates


df_erosivity_5['date'] = _event_date(df_erosivity_5['event_start'])
df_erosivity_15['date'] = _event_date(df_erosivity_15['event_start'])


def _make_cmp(df_rist, df_py):
    """Date-matched merge; RIST one row per day, pyEr summed per day."""
    df_daily = df_py.groupby('date', as_index=False)['erosivity_US'].sum()
    return pd.merge(
        df_rist[['date', 'RIST_EI30']], df_daily, on='date', how='inner'
    )


df_cmp30 = _make_cmp(df_rist_imax30, df_erosivity_5)
df_cmp15 = _make_cmp(df_rist_imax15, df_erosivity_15)


def _print_only_in_py(df_py, df_rist, label):
    """Print events present in pyErosivity but absent from RIST."""
    rist_dates = set(df_rist['date'])
    py_dates = set(df_py['date'])
    depth_by_date = df_py.groupby('date')['event_depth'].sum()
    only_py = sorted(py_dates - rist_dates)
    print(f"\nEvents in pyErosivity ({label}) but NOT in RIST, by year:")
    by_year = {}
    for d in only_py:
        by_year.setdefault(d.year, []).append(d)
    for yr in sorted(by_year):
        entries = ', '.join(
            f"{d} ({depth_by_date.loc[d]:.1f} mm)"
            for d in by_year[yr]
        )
        print(f"  {yr} ({len(by_year[yr])}): {entries}")
    if not only_py:
        print("  (none)")


_print_only_in_py(df_erosivity_5, df_rist_imax30, 'IMax30')
_print_only_in_py(df_erosivity_15, df_rist_imax15, 'IMax15')
print(
    "  NOTE: RIST operates internally in inches with unknown rounding.\n"
    "  Borderline events round to exactly the threshold in RIST and are\n"
    "  excluded (e.g. 12.8 mm → 0.504 in → 0.50 in = 12.7 mm)."
)

all_years = list(range(int(slice_year_from), int(slice_year_to) + 1))

stats_30 = get_mean_annual_stats(
    df_erosivity_5, year_col='event_start', ei30_col='erosivity_US',
    depth_col='event_depth', intensity_col=imax_col,
    all_years=all_years,
)
stats_15 = get_mean_annual_stats(
    df_erosivity_15, year_col='event_start', ei30_col='erosivity_US',
    depth_col='event_depth', intensity_col=imax_col_15,
    all_years=all_years,
)


def _rist_agg(df_rist, col, agg):
    s = df_rist.groupby(df_rist.index.year)[col].agg(agg)
    return s.reindex(all_years, fill_value=0 if agg != 'mean' else None)


r30_ei30 = _rist_agg(df_rist_imax30, 'RIST_EI30', 'sum')
r30_n = _rist_agg(df_rist_imax30, 'RIST_EI30', 'count')
r30_depth = _rist_agg(df_rist_imax30, 'PRECIP', 'mean')
r30_imax = _rist_agg(df_rist_imax30, 'MAX_30', 'mean').dropna()

r15_ei30 = _rist_agg(df_rist_imax15, 'RIST_EI30', 'sum')
r15_n = _rist_agg(df_rist_imax15, 'RIST_EI30', 'count')
r15_depth = _rist_agg(df_rist_imax15, 'PRECIP', 'mean')
r15_imax = _rist_agg(df_rist_imax15, 'MAX_15', 'mean').dropna()

print(
    f"\nEvents — IMax30: RIST {len(df_rist_imax30)} | "
    f"pyEr {len(df_erosivity_5)}"
)
print(
    f"Events — IMax15: RIST {len(df_rist_imax15)} | "
    f"pyEr {len(df_erosivity_15)}"
)
hdr = (
    f"{'':30s} {'RIST 30':>10s} {'pyEr 30':>10s}"
    f" {'RIST 15':>10s} {'pyEr 15':>10s}"
)
print(f"\nMean annual statistics ({slice_year_from}–{slice_year_to}):")
print(hdr)
print(
    f"{'N events / yr':30s} "
    f"{r30_n.mean():>10.1f} {stats_30['n_events']['mean']:>10.1f}"
    f" {r15_n.mean():>10.1f} {stats_15['n_events']['mean']:>10.1f}"
)
print(
    f"{'Mean event depth [mm]':30s} "
    f"{r30_depth.mean():>10.1f} {stats_30['depth']['mean']:>10.1f}"
    f" {r15_depth.mean():>10.1f} {stats_15['depth']['mean']:>10.1f}"
)
print(
    f"{'Mean event Imax [mm/h]':30s} "
    f"{r30_imax.mean():>10.1f} {stats_30['intensity']['mean']:>10.1f}"
    f" {r15_imax.mean():>10.1f} {stats_15['intensity']['mean']:>10.1f}"
)
print(
    f"{'R-factor [MJ mm/ha/h/yr]':30s} "
    f"{r30_ei30.mean():>10.1f} {stats_30['erosivity']['mean']:>10.1f}"
    f" {r15_ei30.mean():>10.1f} {stats_15['erosivity']['mean']:>10.1f}"
)

if save_results:
    df_erosivity_5.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_5min_imax30.parquet.gzip"
        ),
        compression="gzip",
    )
    df_erosivity_15.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_5min_imax15.parquet.gzip"
        ),
        compression="gzip",
    )


def _scatter_panel(ax, df_cmp, title):
    x = df_cmp['RIST_EI30'].values
    y = df_cmp['erosivity_US'].values
    bias = float(np.mean(y - x))
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot
    ax.scatter(x, y, alpha=0.4, s=18, color='steelblue',
               label=f'events (n={len(df_cmp)})')
    lim = max(x.max(), y.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='1:1')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('RIST EI30 [MJ mm ha⁻¹ h⁻¹]')
    ax.set_ylabel('pyErosivity EI30 [MJ mm ha⁻¹ h⁻¹]')
    ax.set_title(title, fontsize=11)
    ax.text(
        0.05, 0.95,
        f"Bias = {bias:+.2f}\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes, fontsize=9,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


fig, axes = plt.subplots(1, 2, figsize=(12, 6))
_scatter_panel(
    axes[0], df_cmp30,
    f"{station_num} — IMax30 (thr={thr_imax30} mm/h)",
)
_scatter_panel(
    axes[1], df_cmp15,
    f"{station_num} — IMax15 (thr={thr_imax15} mm/h)",
)
fig.suptitle(
    'EI30: pyErosivity vs RIST (date-matched)',
    fontsize=13, fontweight='bold',
)
fig.tight_layout()

if save_results:
    fig.savefig(
        os.path.join(_FIG, 'fig02_EI30_scatter.jpeg'),
        format='jpeg', dpi=300,
    )
plt.show()
