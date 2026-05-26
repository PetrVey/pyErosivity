# -*- coding: utf-8 -*-
"""
Benchmarks pyErosivity against RIST for 5-min precipitation data.

The RIST was run with the strict single intensity criterion
(IMax30 >= 12.7 mm/h), so this script uses the same setting
(use_both_thresholds=False) to allow a direct event-by-event comparison.

WARNING — SINGLE THRESHOLD VS FULL RUSLE
-----------------------------------------
The full RUSLE definition (Wischmeier & Smith 1978; Renard et al. 1997)
uses TWO criteria — erosive if EITHER:
    (i)  accumulated event depth >= 12.7 mm, OR
    (ii) maximum 15-min depth   >= 6.35 mm

This script intentionally applies only criterion (ii) as an intensity
threshold to match RIST output.
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
data_rist = lines[3:]
df_erosivity_5min_RIST = pd.DataFrame(
    [line.split() for line in data_rist], columns=header
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
print(f"Calculation time: {elapsed_time:.2f} s")

#%%
# == # == # == # == # == # == # == # == # == # ==
# == # == # COMPARISON # == # COMPARISON # == # ==
# == # == # == # == # == # == # == # == # == # ==

df_erosivity_5min_RIST['date'] = df_erosivity_5min_RIST.index.date
df_erosivity_5['date'] = df_erosivity_5['event_start'].dt.date

df_erosivity_5min_RIST = df_erosivity_5min_RIST.rename(
    columns={"EI30": "RIST_EI30"}
)
df_erosivity_5 = df_erosivity_5.rename(
    columns={"erosivity_US": "erosivity_US_5"}
)

print("\n--- RIST events ---")
print(df_erosivity_5min_RIST[['RIST_EI30']].to_string())
print("\n--- pyErosivity events ---")
print(
    df_erosivity_5[['event_start', 'event_end', 'erosivity_US_5']]
    .to_string(index=False)
)

df_cmp = pd.merge(
    df_erosivity_5min_RIST[['date', 'RIST_EI30']],
    df_erosivity_5[['date', 'erosivity_US_5']],
    on='date', how='inner',
)

x_col = 'erosivity_US_5'
y_col = 'RIST_EI30'
x_label = "pyErosivity Re_5min [MJ mm ha⁻¹ h⁻¹]"
y_label = "RIST EI30 [MJ mm ha⁻¹ h⁻¹]"

X = df_cmp[[x_col]]
y = df_cmp[y_col]
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

r2 = model.score(X, y)
rmse = sqrt(mean_squared_error(df_cmp[x_col], df_cmp[y_col]))
bias = np.mean(df_cmp[y_col] / df_cmp[x_col])

axs_limit = 1400

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df_cmp[x_col], df_cmp[y_col], color='black', s=15)
ax.plot(
    df_cmp[x_col], y_pred,
    color='blue', label=f"Regression (R²={r2:.2f})",
)
ax.plot(
    np.linspace(0, axs_limit, 100),
    np.linspace(0, axs_limit, 100),
    color='black', linestyle='--', linewidth=0.8,
    label='1:1 line',
)
ax.text(
    0.05, 0.95,
    f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nMBR: {bias:.2f}",
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='left',
    bbox=dict(
        facecolor='white', edgecolor='black',
        boxstyle='round,pad=0.5',
    ),
)
ax.set_xlim(0, axs_limit)
ax.set_ylim(0, axs_limit)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_xticks(np.arange(0, axs_limit + 1, 200))
ax.set_yticks(np.arange(0, axs_limit + 1, 200))
ax.legend()
ax.grid(True)
fig.suptitle(
    f"{station_num} — pyErosivity vs RIST (IMax30 only)",
    fontsize=13, fontweight='bold',
)

if save_results:
    fig.savefig(
        os.path.join(_FIG, 'fig01_Re_RIST_vs_pyErosivity.jpeg'),
        format='jpeg', dpi=300,
    )
plt.show()

print(
    f"Events — RIST: {len(df_erosivity_5min_RIST)} | "
    f"pyErosivity: {len(df_erosivity_5)} | "
    f"matched: {len(df_cmp)}"
)
