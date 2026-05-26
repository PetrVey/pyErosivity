# -*- coding: utf-8 -*-
"""
Benchmarks pyErosivity against RIST and compares 5-min vs 60-min
erosivity.

The RIST was run only with the strict single-threshold
(IMax30 = 12.7 mm/h), not the full dual-criterion RUSLE.

This script uses ONLY the single intensity criterion
(use_both_thresholds=False):
    IMax30 >= intensity_threshold [mm/h]

WARNING — SINGLE THRESHOLD VS FULL RUSLE
-----------------------------------------
The full RUSLE definition (Wischmeier & Smith 1978; Renard et al. 1997)
uses TWO criteria — erosive if EITHER:
    (i)  accumulated event depth >= 12.7 mm, OR
    (ii) maximum 15-min depth   >= 6.35 mm

Criterion (ii) is a depth criterion, not a sustained intensity.
Converting it to IMax15 >= 25.4 mm/h assumes 6.35 mm falls uniformly
over the full 15 min, which is rarely the case in practice.

Williams & Sheridan (1991) were among the first to show that coarser
measurement intervals systematically underestimate EI30; for 60-min
data they set I30 equal to the maximum hourly depth (I30 = I60) as a
pragmatic resolution adaptation.
Fischer et al. (2018) extended this line of work by lowering the IMax30
threshold for coarser resolutions, assuming the 6.35 mm peak
concentrates within a sub-window while the rest of the hour is nearly
dry. The optimisation in this script follows the same logic: find the
IMax30 threshold that reproduces the 5-min event count, implicitly
correcting for resolution-induced under-detection.
"""

import os
import time
from math import sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from pyErosivity import remove_incomplete_years
from pyErosivity import get_events
from pyErosivity import remove_short
from pyErosivity import get_events_values
from pyErosivity import get_only_erosivity_events
from pyErosivity import find_optimal_thr_imax30

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, '..', 'res')
_OUT = os.path.join(_HERE, '..', 'out')
_FIG = os.path.join(_HERE, '..', 'fig')

#%%
save_results = False
station_num = "VE_0091"
slice_year_from = "2012"
slice_year_to = "2020"

# Read RIST output for comparison
file_path = os.path.join(
    _RES, f"RIST_{station_num}_Erosive_Events_5minutes.txt"
)
with open(file_path, "r") as file:
    lines = file.readlines()
header = lines[0].split()
data = lines[3:]
df_erosivity_5min_RIST = pd.DataFrame(
    [line.split() for line in data], columns=header
)
df_erosivity_5min_RIST["Date"] = pd.to_datetime(
    df_erosivity_5min_RIST["Date"], format="%m/%d/%Y"
)
df_erosivity_5min_RIST = df_erosivity_5min_RIST.set_index("Date")
df_erosivity_5min_RIST = df_erosivity_5min_RIST.astype(
    {col: 'float64' for col in df_erosivity_5min_RIST.columns[:]}
)
df_erosivity_5min_RIST = df_erosivity_5min_RIST[
    slice_year_from:slice_year_to
]
if save_results:
    df_erosivity_5min_RIST.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_RIST_5min.parquet.gzip"
        ),
        compression="gzip",
    )

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # 5 min data # == # == # == # == #
# == # == # == # 30 minutes intensity # ==  == #
separation = 6           # Dry spell between events [hours]
                         # (Wischmeier and Smith, 1958, 1978)
durations = [30]         # Accumulation windows [min]
min_rain = 0.1           # Minimum rain depth [mm]
min_ev_dur = 30          # Minimum event duration [min]
time_resolution = 5.0    # Dataset time resolution [min]
name_col = "vals"        # Column with precipitation data
use_both_thresholds = False  # Single threshold to match RIST setup
thr_imax30 = 12.7        # Wischmeier IMax30 threshold [mm/h]
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

data = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_5min_newflag.parguqet.gzip")
)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.loc[slice_year_from:slice_year_to]

start_time = time.time()

data.loc[data['flag'] > 0, name_col] = np.nan
data.loc[data[name_col] < min_rain, name_col] = 0

data, time_resolution_check = remove_incomplete_years(
    data, name_col, nan_to_zero=True, tolerance=0.1
)

if time_resolution_check == time_resolution:
    print("Given time resolution is matching with dataset")
else:
    print("There might be inconsistency in dataset time resolution")

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

idx_events = get_events(
    data=df_arr,
    dates=df_dates,
    separation=separation,
    min_rain=min_rain,
    name_col=name_col,
    check_gaps=False,
)

arr_vals, arr_dates, n_events_per_year = remove_short(
    idx_events,
    time_resolution=time_resolution,
    min_ev_dur=min_ev_dur,
)

dict_events = get_events_values(
    data=df_arr,
    dates=df_dates,
    arr_dates_oe=arr_dates,
    durations=durations,
    time_resolution=time_resolution,
)

df_erosivity_all_events = dict_events["30"].copy()
df_erosivity_5 = get_only_erosivity_events(
    df_erosivity_all_events,
    use_both_thresholds=use_both_thresholds,
    intensity_threshold=thr_imax30,
)

if save_results:
    df_erosivity_5.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_5min.parquet.gzip"
        ),
        compression="gzip",
    )

elapsed_time = time.time() - start_time
print(
    f"Calculation of erosivity of 5min resolution : "
    f"{elapsed_time:.2f} seconds"
)

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # 60 min data # == # == # == # == #
# == # == # == # 60 minutes intensity # ==  == #
separation = 6           # Dry spell between events [hours]
                         # (Wischmeier and Smith, 1958, 1978)
durations = [60]         # Accumulation windows [min]
                         # For Δt = 60 min, I30 = I60 = max hourly
                         # depth (Williams and Sheridan, 1991)
min_rain = 0.1           # Minimum rain depth [mm]
min_ev_dur = 30          # Minimum event duration [min]
time_resolution = 60.0   # Dataset time resolution [min]
name_col = "vals"        # Column with precipitation data
use_both_thresholds = False  # Single threshold to match RIST setup
# Fischer et al. (2018) doi:10.5194/hess-22-6505-2018
thr_imax30 = 5.79        # German IMax30 for 60-min data [mm/h]
temporal_scale_factor = 1.9  # Temporal scaling (Fischer et al. 2018)
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

data = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_1h_flag.parguqet.gzip")
)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.loc[slice_year_from:slice_year_to]

start_time = time.time()

data.loc[data['flag'] > 0, name_col] = np.nan
data.loc[data[name_col] < min_rain, name_col] = 0

data, time_resolution_check = remove_incomplete_years(
    data, name_col, nan_to_zero=True, tolerance=0.1
)

if time_resolution_check == time_resolution:
    print("Given time resolution is matching with dataset")
else:
    print("There might be inconsistency in dataset time resolution")

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

idx_events = get_events(
    data=df_arr,
    dates=df_dates,
    separation=separation,
    min_rain=min_rain,
    name_col=name_col,
    check_gaps=False,
)

arr_vals, arr_dates, n_events_per_year = remove_short(
    idx_events,
    time_resolution=time_resolution,
    min_ev_dur=min_ev_dur,
)

dict_events = get_events_values(
    data=df_arr,
    dates=df_dates,
    arr_dates_oe=arr_dates,
    durations=durations,
    time_resolution=time_resolution,
)

# For 60-min data IMax30 = IMax60 = single hourly reading
# (Williams & Sheridan 1991)
df_erosivity_all_events_60 = dict_events["60"].copy()

df_erosivity_60 = get_only_erosivity_events(
    df_erosivity_all_events_60,
    use_both_thresholds=use_both_thresholds,
    intensity_threshold=thr_imax30,
)
n_events_german_60 = len(df_erosivity_60)

df_erosivity_60["erosivity_US_adj"] = (
    df_erosivity_60["erosivity_US"] * temporal_scale_factor
)

if save_results:
    df_erosivity_60.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_60min.parquet.gzip"
        ),
        compression="gzip",
    )

elapsed_time = time.time() - start_time
print(
    f"Calculation of erosivity of 60min resolution : "
    f"{elapsed_time:.2f} seconds"
)

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # OPTIMIZE thr_imax30 FOR 60min DATA # == # ==
# == # == # == # == # == # == # == # == # == # ==
target_count = len(df_erosivity_5)
thr_opt_60, achieved_count_60, residual = find_optimal_thr_imax30(
    df_erosivity_all_events_60,
    target_count,
    use_both_thresholds=use_both_thresholds,
)

print(f"Target event count  (5-min data) : {target_count}")
print(f"Optimal thr_imax30  (60-min data): {thr_opt_60:.4f} mm/h")
print(f"Achieved event count             : {achieved_count_60}")
print(f"Residual |achieved - target|     : {residual}")
if residual == 0:
    print("Exact match found.")
else:
    print("No exact match possible (step-function gap).")

df_erosivity_60 = get_only_erosivity_events(
    df_erosivity_all_events_60,
    use_both_thresholds=use_both_thresholds,
    intensity_threshold=thr_opt_60,
)
df_erosivity_60["erosivity_US_adj"] = (
    df_erosivity_60["erosivity_US"] * temporal_scale_factor
)
if save_results:
    df_erosivity_60.to_parquet(
        os.path.join(
            _OUT, f"{station_num}_erosivity_60min.parquet.gzip"
        ),
        compression="gzip",
    )

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # COMPARISON # == # COMPARISON # == # ==
# == # == # PLOTS # == # PLOTS # == # == # == # ==
df_erosivity_5min_RIST['date'] = df_erosivity_5min_RIST.index.date
df_erosivity_5['date'] = df_erosivity_5['event_start'].dt.date
df_erosivity_60['date'] = df_erosivity_60['event_start'].dt.date

df_erosivity_60 = df_erosivity_60.rename(columns={
    "erosivity_US_adj": "erosivity_US_adj_60",
    "erosivity_US": "erosivity_US_60",
})
df_erosivity_5min_RIST = df_erosivity_5min_RIST.rename(
    columns={"EI30": "RIST_EI30"}
)
df_erosivity_5 = df_erosivity_5.rename(
    columns={"erosivity_US": "erosivity_US_5"}
)

df_cmp_RIST_5 = pd.merge(
    df_erosivity_5min_RIST[['date', 'RIST_EI30']],
    df_erosivity_5[['date', 'erosivity_US_5']],
    on='date', how='inner',
)
df_cmp_60_5 = pd.merge(
    df_erosivity_5[['date', 'erosivity_US_5']],
    df_erosivity_60[['date', 'erosivity_US_60']],
    on='date', how='inner',
)
df_cmp_60adj_5 = pd.merge(
    df_erosivity_5[['date', 'erosivity_US_5']],
    df_erosivity_60[['date', 'erosivity_US_adj_60']],
    on='date', how='inner',
)

if save_results:
    df_comparison = pd.merge(
        df_cmp_RIST_5,
        df_erosivity_60[
            ['date', 'erosivity_US_60', 'erosivity_US_adj_60']
        ],
        on='date', how='inner',
    )
    df_comparison.to_parquet(
        os.path.join(
            _OUT,
            f"{station_num}_erosivity_comparison.parquet.gzip",
        ),
        compression="gzip",
    )

x = 'erosivity_US_5'
x_label = "Re_5min"
pairs = [
    (df_cmp_RIST_5,   'RIST_EI30',          'Re_RIST_5min'),
    (df_cmp_60_5,     'erosivity_US_60',     'Re_60min_opt'),
    (df_cmp_60adj_5,  'erosivity_US_adj_60', 'Re_60min_opt_corr'),
]

axs_limit = 1400
fig, axs = plt.subplots(2, 2, figsize=(8.5, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
axs[1, 1].axis('off')

legend_elements = []
scatter_plot = None
colors = ['blue', 'green', 'red']

for i, (df_pair, y_col, label) in enumerate(pairs):
    ax = axs[i // 2, i % 2]

    scatter_plot = ax.scatter(
        df_pair[x], df_pair[y_col],
        color='black', label="Erosivity in [MJ*mm/ha*hr]",
    )

    X = df_pair[[x]]
    y = df_pair[y_col]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    reg_line, = ax.plot(
        df_pair[x], y_pred,
        color=colors[i], label=f"Regression: {label}",
    )
    identity_line, = ax.plot(
        np.linspace(0, axs_limit, 100),
        np.linspace(0, axs_limit, 100),
        color='black', linestyle='--', linewidth=0.8,
        label='Identity Line',
    )

    r2 = model.score(X, y)
    rmse = sqrt(mean_squared_error(df_pair[x], df_pair[y_col]))
    bias = np.mean(df_pair[y_col] / df_pair[x])

    ax.text(
        0.05, 0.95,
        f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nMBR: {bias:.2f}",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='left',
        color='black',
        bbox=dict(
            facecolor='white', edgecolor='black',
            boxstyle='round,pad=0.5',
        ),
    )
    ax.set_xlim(0, axs_limit)
    ax.set_ylim(0, axs_limit)
    ax.set_xlabel(f'{x_label} [MJ*mm/ha*hr]')
    ax.set_ylabel(f'{label} [MJ*mm/ha*hr]')
    ax.set_xticks(np.arange(0, axs_limit + 1, 200))
    ax.set_yticks(np.arange(0, axs_limit + 1, 200))
    ax.grid(True)

    existing_labels = [e[0].get_label() for e in legend_elements]
    if identity_line.get_label() not in existing_labels:
        legend_elements.append((identity_line, 'Identity Line'))
    if reg_line not in [e[0] for e in legend_elements]:
        legend_elements.append((reg_line, f"Regression: {label}"))

fig.legend(
    [scatter_plot] + [e[0] for e in legend_elements],
    ['Erosivity Re in [MJ*mm/ha*hr]'] + [e[1] for e in legend_elements],
    loc='center', fontsize=14, frameon=False,
    title="Legend", title_fontsize=16,
    bbox_to_anchor=(0.7, 0.3),
)
fig.suptitle(
    f"{station_num} Erosivity Re in [MJ*mm/ha*hr] for events",
    fontsize=14, fontweight='bold', y=0.93,
)
if save_results:
    fig.savefig(
        os.path.join(_FIG, 'fig00_Re_comparison.jpeg'),
        format='jpeg', dpi=300,
    )
plt.show()

lengths = [
    ('RE 5 min from RIST', len(df_erosivity_5min_RIST)),
    ('Re 5min', len(df_erosivity_5)),
    (
        f'Re 60min\n(German thr_imax30 = {thr_imax30})',
        n_events_german_60,
    ),
    (
        f'Re 60min\n(Optimised thr_imax30 = {thr_opt_60:.4f})',
        achieved_count_60,
    ),
]

fig, ax = plt.subplots(figsize=(6, 3))
ax.axis('off')

table_data = [[name, length] for name, length in lengths]
table = ax.table(
    cellText=table_data, loc='center',
    colLabels=['DataFrame', 'Length'],
    cellLoc='center', colColours=['#f5f5f5', '#f5f5f5'],
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_fontsize(14)
        cell.set_text_props(weight='bold')
    cell.set_height(0.3)
    cell.set_width(0.7)

if save_results:
    fig.savefig(
        os.path.join(_FIG, 'fig00_RE_datasets_lenght.jpeg'),
        format='jpeg', dpi=300, bbox_inches='tight',
    )
plt.show()
