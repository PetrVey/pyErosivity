# -*- coding: utf-8 -*-
"""
Bootstrap uncertainty analysis for annual erosivity — CPM use case.

Data: station VE_0091, ETH CPM Historical 1996-2005 (10 years), 1-hour res.
Uses a pre-defined bootstrap sample sequence (randy.txt) shared across
datasets for a fair comparison between climate model ensemble members.

Reference: Dallan et al. (2023) https://doi.org/10.5194/hess-27-1133-2023
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyErosivity import remove_incomplete_years
from pyErosivity import get_events
from pyErosivity import remove_short
from pyErosivity import get_events_values
from pyErosivity import compute_erosivity
from pyErosivity import get_only_erosivity_events
from pyErosivity import bootstrapping_erosivity_CPM_60min

station_num = "VE_0091"

# == # == # == # == # == # == # == # == # == # ==
# == # == # SETTING # == # == # SETTING # == # ==
separation = 6            # Minimum dry-spell between events [hours]
min_rain = 0.1            # Drizzle threshold [mm]
min_ev_dur = 30           # Minimum event duration [min]
time_resolution = 60.0    # Data time step [min]
name_col = "pr_new"       # Precipitation column name
use_both_thresholds = False
thr_imax30 = 5.79         # Adjusted IMax60 threshold for 60-min data [mm/h]
temporal_scale_factor = 1.9
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # ==

# Load data
data = pd.read_csv(f"res/{station_num}_ETH_hist.csv")
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')

# Load pre-defined bootstrap sample sequences, shape (1000, 10)
randy = np.loadtxt('res/randy.txt', delimiter=',')
randy = randy.T.astype(np.int32)

# %% === STAGE 1: Prepare data ===

# Zero out drizzle
data.loc[data[name_col] < min_rain, name_col] = 0

data, time_resolution_check = remove_incomplete_years(
    data, name_col, nan_to_zero=True, tolerance=0.1
)
if time_resolution_check == time_resolution:
    print("Time resolution matches dataset")
else:
    print("Warning: time resolution mismatch")

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

# %% === STAGE 2: Extract and filter events ===

arr_dates = get_events(
    data=df_arr, dates=df_dates,
    separation=separation, min_rain=min_rain,
    check_gaps=False,
)

_, arr_dates, _ = remove_short(
    arr_dates, time_resolution=time_resolution, min_ev_dur=min_ev_dur,
)

df_events = get_events_values(
    data=df_arr, dates=df_dates,
    arr_dates_oe=arr_dates, time_resolution=time_resolution,
)

df_events = compute_erosivity(df_events)

df_erosivity = get_only_erosivity_events(
    df_events,
    imax_col='imax_60',
    intensity_threshold=thr_imax30,
    use_both_thresholds=use_both_thresholds,
)

# Optional temporal scale correction for 60-min data
df_erosivity = df_erosivity.copy()
df_erosivity['erosivity_US_adj'] = (
    df_erosivity['erosivity_US'] * temporal_scale_factor
)

# %% === STAGE 3: Annual statistics of population ===

years = df_erosivity['event_start'].dt.year
yearly_stats = df_erosivity.assign(_year=years).groupby('_year').agg(
    N_events=('event_depth', 'count'),
    mean_intensity_per_hour=('imax_60', 'mean'),
    mean_event_depth=('event_depth', 'mean'),
    sum_erosivity_US_adj=('erosivity_US_adj', 'sum'),
).reset_index().rename(columns={'_year': 'year'})

overall_means = yearly_stats[[
    'N_events', 'mean_intensity_per_hour',
    'mean_event_depth', 'sum_erosivity_US_adj',
]].mean()
overall_means.index = [
    'mean_annual_events', 'mean_annual_Imax',
    'mean_rain_depth', 'average_annual_erosivity',
]
print("Annual statistics of population")
print(overall_means)

# %% === STAGE 4: Bootstrap uncertainty ===

df_bootstrap_summary = bootstrapping_erosivity_CPM_60min(
    df_erosivity,
    randy=randy,
    imax_col='imax_60',
    erosivity_col='erosivity_US_adj',
)

# %% === STAGE 5: Plot ===

variables = [
    'mean_annual_events',
    'mean_annual_Imax',
    'mean_rain_depth',
    'average_annual_erosivity',
]

ylims = [
    (22, 30),
    (8, 12),
    (40, 54),
    (4000, 6000),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]
    ax.boxplot(df_bootstrap_summary[var], vert=True)
    ax.scatter(
        1, overall_means[var],
        color='red', s=80, label='Original value', marker='D', zorder=5,
    )
    ax.set_title(var.replace('_', ' ').capitalize())
    ax.set_xticks([1])
    ax.set_xticklabels(['Bootstrap'])
    ax.set_ylim(ylims[i])
    ax.grid(True, linestyle='--', alpha=0.5)
    if i == 0:
        ax.legend()

plt.suptitle(
    'Bootstrap Distributions vs Original Annual Statistics',
    fontsize=18, y=1,
)
plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.show()
