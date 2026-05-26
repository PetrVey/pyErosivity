# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:16:59 2025

@author: Petr

Script shows how to apply uncertainty analysis from bootstrapping with 1000 samples.

Data used in this example are from station co-located grid from ETH CPM Historical period.
Data are from 1996-2005, so 10 years slice.
We again use same station VE_0091

BOOTSTRAPPING IS DONE AT STAGE 4 (with loaded prepared bootstrap samples sequances)
We use preprepared boostrap sampple sequance which has been used in previous research.
The sequence is done by randomizing order of 10 available years 1000x
#  Dallan et al., 2023 
#  How well does a convection-permitting regional climate model represent
#  the reverse orographic effect of extreme hourly precipitation?, 
#  https://doi.org/10.5194/hess-27-1133-2023


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.pyErosivity import remove_incomplete_years
from src.pyErosivity import get_events 
from src.pyErosivity import remove_short 
from src.pyErosivity import get_events_values 
from src.pyErosivity import get_only_erosivity_events
from src.pyErosivity import boostrapping_erosivity_CPM_60min


# We use VE_0091 station data in resampled hourly resolution as an example
station_num = "VE_0091"

# == # == # == # == # == # == # == # == # == # == 
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # 60 min data # == # == # == # == #
# == # == # == # 60 minutes intensity # ==  == #
separation = 6           # Separation time between idependent storms -> dryspell between two events [hours]
                         # --> Rain breaks of at least 6 h (Wischmeier and Smith, 1958, 1978)
durations = [60]         # List of durations for which we calculate rainfall depth [min]
                         #For Δt = 60 min, I30 was set equal to the maximum 60-min accumulated depth (Williams and Sheridan, 1991).
min_rain = 0.1           # Minimum threshold for rain depth -> climate models have a drizzle problem [mm]
min_ev_dur = 30          # Minimum event duration [min]
time_resolution = 60.0    # Time resolution of dattaset [min]
name_col = "pr_new"      # Name of column containing data to extract
use_both_thresholds = False # True Defined erosivity event as intensity >= threshold1 & accum_prec >= threshold2
                            # False Defined erosivity event as intensity >= threshold1
                            # --> here we have false cause we set RIST for single threshold on intensity                      
#Following setting source https://hess.copernicus.org/articles/22/6505/2018/
thr_imax30 = 5.79       #  adjusted threshold for imax30 due to lower resolution data 
#temporal scale factor is from German study, it is better to estimate own scaling factor that will match data
temporal_scale_factor = 1.9 # https://hess.copernicus.org/articles/22/6505/2018/
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # == 


# Load the data from CSV
data = pd.read_csv(f"res/{station_num}_ETH_hist.csv")
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')

# Load randy file with prepapred boostrap samples sequances
randy = np.loadtxt('res/randy.txt', delimiter=',')
randy = randy.T  # Transpose to get shape (1000, 10)
randy = randy.astype(np.int32)

#%% # === STAGE 1: Calculate event staitics of population ===

# Push values belows 0.1 to 0 in prec due to drizzle problem
data.loc[data[name_col] < min_rain, name_col] = 0

# Remove incomplete years from dataset (if needed, we usually clean years that has 10% of missing values)
data, time_resolution_check = remove_incomplete_years(data,
                                                      name_col,
                                                      nan_to_zero=True, 
                                                      tolerance=0.1)

# Just check if time resolution in dataset is consitent with the one we setup
if time_resolution_check == time_resolution:  
    print("Given time resolution is matching with dataset")
else: 
    print("There might be inconsistency in dataset time resolution")


#Transfer data from pandas to numpy array due to time efficiency
df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

#extract indexes of events
#these are time-wise indexes => returns list of np arrays with np.timeindex
idx_events=get_events(data = df_arr,
                      dates = df_dates, 
                      separation = separation,
                      min_rain = min_rain,
                      name_col = name_col,  
                      check_gaps = False)
    
#get events by removing too short events
#returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
arr_vals,arr_dates,n_events_per_year=remove_short(idx_events, 
                                                    time_resolution=time_resolution, 
                                                    min_ev_dur=min_ev_dur)

#assign events events values by given durations, values are in depth per duration, NOT in intensity mm/h
dict_events= get_events_values(data=df_arr,
                               dates=df_dates, 
                               arr_dates_oe=arr_dates,
                               durations=durations,
                               time_resolution=time_resolution)

# Here example for 60 minutes, so it extract 60 out of this dict_events
df_erosivity_all_events_60 = dict_events["60"].copy()

# Get erosivity events with different intesnity threshold, this threshold should be optimized based on needs.
df_erosivity_60 = get_only_erosivity_events(df_erosivity_all_events_60, 
                                            use_both_thresholds=use_both_thresholds, 
                                            intensity_threshold=thr_imax30)

# If temporal adjustment for resolution is needed, again, this scale factor should be better determined by needs
df_erosivity_60["erosivity_US_adj"] = df_erosivity_60["erosivity_US"] * temporal_scale_factor



#%% # === STAGE 2: Calculate Yearly staitics from population ===

# Ensure 'event_peak' is datetime
df_erosivity_60['event_peak'] = pd.to_datetime(df_erosivity_60['event_peak'])

# Group by year extracted from 'event_peak'
# for this we use aggregate function of pandas to get more statistics out of one dataframe
yearly_stats = df_erosivity_60.groupby(df_erosivity_60['event_peak'].dt.year).agg(
    N_events=('event_peak', 'count'),
    mean_intensity_per_hour=('intensity_per_hour', 'mean'),
    mean_prec_accum=('prec_accum', 'mean'),
    sum_erosivity_US_adj=('erosivity_US_adj', 'sum')
).reset_index()

# Rename the index column to 'year' for clarity
yearly_stats = yearly_stats.rename(columns={'event_peak': 'year'})

#%% # === STAGE 3: Calculate annual means/sums of population ===
#Out of yearly statistic, we can calculce annual averages/sums
# 1. Mean annual number of erosive events,
# 2. Mean Maximum 60-minute rainfall intensity (Imax),
# 3. Mean rainfall depth per erosive event, and
# 4. Average annual rainfall erosivity.

# Mean across years
overall_means = yearly_stats[['N_events', 'mean_intensity_per_hour', 'mean_prec_accum', 'sum_erosivity_US_adj']].mean()

# Rename for clarity (optional)
overall_means.index = [
    'mean_annual_events',
    'mean_annual_Imax',
    'mean_rain_depth',
    'average_annual_erosivity'
]

# Display the result
print("Annual statitics of population")
print(overall_means)

#%% # === STAGE 4: Bootstrapp the events to accomodate for uncertainty ===

# BE SURE THAT RANDY IS LOADED AND HAS SHAPE (1000, 10), or where each row is sequance of years.
df_bootstrap_summary = boostrapping_erosivity_CPM_60min(df_erosivity_60,
                                                        randy=randy)

#%% # === STAGE 5: Bootstrapp the events to accomodate for uncertainty ===

# Define variables and y-axis limits
variables = [
    'mean_annual_events',
    'mean_annual_Imax',
    'mean_rain_depth',
    'average_annual_erosivity'
]

ylims = [
    (22, 30),       # for mean_annual_events
    (8, 12),        # for mean_annual_Imax
    (40, 54),       # for mean_rain_depth
    (4000, 6000)    # for average_annual_erosivity
]

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()
 
for i, var in enumerate(variables):
    ax = axes[i]
    # Boxplot from bootstrap samples
    ax.boxplot(df_bootstrap_summary[var], vert=True)

    # Overlay red point for original sample
    ax.scatter(1, overall_means[var], color='red', s=80, label='Original value', marker='D', zorder=5)

    # Set plot title, labels, grid
    ax.set_title(var.replace('_', ' ').capitalize())
    ax.set_xticks([1])
    ax.set_xticklabels(['Bootstrap'])
    ax.set_ylim(ylims[i])
    ax.grid(True, linestyle='--', alpha=0.5)

    if i == 0:
        ax.legend()

# Add a clearer super-title and space it higher
plt.suptitle('Comparison of Bootstrap Distributions and Original Annual Statistics', 
             fontsize=18, y=1)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.90)  # reserve space for suptitle
plt.show()
