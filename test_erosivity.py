# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:55:27 2024

@author: Petr
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import time
from src.pyErosivity import remove_incomplete_years
from src.pyErosivity import get_events 
from src.pyErosivity import remove_short 
from src.pyErosivity import get_events_values 
from src.pyErosivity import get_only_erosivity_events

#%%
save_results = True
station_num = "VE_0091"
#We sliced data from this year to for consistency with RIST oputput
# This slice was done due to time constrasints in calculation.
slice_year_from = "2012" #
slice_year_to = "2020"

# READ DATA OF RIST TOOL OUTPUT FOR FURTHER COMPARSION
# Path to your text file
file_path = f"res/RIST_{station_num}_Erosive_Events_5minutes.txt"
# Read the file, handling headers and data separately
with open(file_path, "r") as file:
    lines = file.readlines()
# Extract the header row and clean it
header = lines[0].split()  # Use the first line as the header
data = lines[3:]           # Data starts from the 4th line onwards
df_erosivity_5min_RIST = pd.DataFrame([line.split() for line in data], columns=header)
# Convert columns to appropriate data types (float)
df_erosivity_5min_RIST["Date"] = pd.to_datetime(df_erosivity_5min_RIST["Date"], format="%m/%d/%Y")  # Parse dates
df_erosivity_5min_RIST = df_erosivity_5min_RIST.set_index("Date")
df_erosivity_5min_RIST = df_erosivity_5min_RIST.astype({col: 'float64' for col in df_erosivity_5min_RIST.columns[:]})
df_erosivity_5min_RIST = df_erosivity_5min_RIST[slice_year_from:slice_year_to]
if save_results:
    df_erosivity_5min_RIST.to_parquet(f"out/{station_num}_erosivity_RIST_5min.parquet.gzip", compression="gzip")

#%%
# == # == # == # == # == # == # == # == # == # == 
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # 5 min data # == # == # == # == #
# == # == # == # 30 minutes intensity # ==  == #
separation = 6           # Separation time between idependent storms -> dryspell between two events [hours]
                         # rain breaks of at least 6 h (Wischmeier and Smith, 1958, 1978)
                         
durations = [30]         # List of durations for which we calculate rainfall depth [min]
min_rain = 0.1           # Minimum threshold for rain depth -> climate models have a drizzle problem [mm]
min_ev_dur = 30          # Minimum event duration [min]
time_resolution = 5.0    # Time resolution of dattaset [min]
name_col = "vals"        # Name of column containing data to extract
use_both_thresholds = False # True Defined erosivity event as intensity >= threshold1 & accum_prec >= threshold2
                            #False Defined erosivity event as intensity >= threshold1
                            #here we have false cause we set RIST for single threshold on intensity
#Following setting source https://hess.copernicus.org/articles/22/6505/2018/
thr_imax30 = 12.7         #  adjusted threshold for imax30 due to lower resolution data 
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # == 
# Start timer
start_time = time.time()

# Load the data from CSV
data = pd.read_parquet(f"res/{station_num}_5min_newflag.parguqet.gzip")
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.loc[slice_year_from:slice_year_to]

# Treat flagged points as np.nan
# This is becuase we first remove incomplete years and np.nans are pushed to 0 after
data.loc[data['flag'] > 0, 'vals'] = np.nan


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

# Here example for 30 minutes, so it extract 30 out of this dict_events
df_erosivity_all_events = dict_events["30"].copy()
df_erosivity_5 = get_only_erosivity_events(df_erosivity_all_events, 
                                         use_both_thresholds=False,
                                         intensity_threshold=thr_imax30)
# == # == # == # SAVE RESULTS # == # == == # == # 
if save_results:
    df_erosivity_5.to_parquet(f"out/{station_num}_erosivity_5min.parquet.gzip", compression="gzip")

# End timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Calculation of erosivity of 5min resolution : {elapsed_time:.2f} seconds")
#%%
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
name_col = "vals"        # Name of column containing data to extract
use_both_thresholds = False # True Defined erosivity event as intensity >= threshold1 & accum_prec >= threshold2
                            # False Defined erosivity event as intensity >= threshold1
                            # --> here we have false cause we set RIST for single threshold on intensity                      
#Following setting source https://hess.copernicus.org/articles/22/6505/2018/
thr_imax30 = 5.79       #  adjusted threshold for imax30 due to lower resolution data 
#temporal scale factor is from German study, it is better to estimate own scaling factor that will match data
temporal_scale_factor = 1.9 # https://hess.copernicus.org/articles/22/6505/2018/
# == # == # SETTING # == # == # SETTING # == # ==
# == # == # == # == # == # == # == # == # == # == 

# Start timer
start_time = time.time()
# Load the data from CSV
data = pd.read_parquet(f"res/{station_num}_1h_flag.parguqet.gzip")
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data = data.loc[slice_year_from:slice_year_to]


# Treat flagged points as np.nan
# This is becuase we first remove incomplete years and np.nans are pushed to 0 after
data.loc[data['flag'] > 0, 'vals'] = np.nan


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
                                            use_both_thresholds=False, 
                                            intensity_threshold=thr_imax30)

#If temporal adjustment for resolution is needed, again, this scale factor should be better determined by needs
df_erosivity_60["erosivity_US_adj"] = df_erosivity_60["erosivity_US"] * temporal_scale_factor
# == # == # == # SAVE RESULTS # == # == == # == # 
if save_results:
    df_erosivity_60.to_parquet(f"out/{station_num}_erosivity_60min.parquet.gzip", compression="gzip")

# End timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Calculation of erosivity of 60min resolution : {elapsed_time:.2f} seconds")

#%%
# == # == # == # == # == # == # == # == # == # == 
# == # == # COMPARSION # == # COMPARISON # == # ==
# == # == # PLOTS # == # PLOTS # == # ==# ==# ==

# There are a few problems that selected methods do not return same number of erosivity events
# This is due to different approach between RIST , our calculcation wit t=5min and t=60min
# It is absolutely expected behavior and we will work on adjustments later 

df_erosivity_5["erosivity_US"]
df_erosivity_60["erosivity_US_adj"] 

df_erosivity_5min_RIST['date'] = df_erosivity_5min_RIST.index.date  # Convert index to 'yyyy-mm-dd'
df_erosivity_5['date'] = df_erosivity_5['event_start'].dt.date
df_erosivity_60['date'] = df_erosivity_60['event_start'].dt.date
#for clarity sake, we renamed columns
df_erosivity_60 = df_erosivity_60.rename(columns={"erosivity_US_adj" : "erosivity_US_adj_60",
                                                  "erosivity_US" :  "erosivity_US_60"})
df_erosivity_5min_RIST = df_erosivity_5min_RIST.rename(columns={"EI30" : "RIST_EI30"})
df_erosivity_5 = df_erosivity_5.rename(columns={"erosivity_US" : "erosivity_US_5"})



result = pd.merge(df_erosivity_5min_RIST, df_erosivity_5, on='date', how='inner')
# Merge the result with the new dataset
final_result = pd.merge(result, df_erosivity_60 , on='date', how='inner')
df_comparison = final_result[['date', 'RIST_EI30', 'erosivity_US_5', 'erosivity_US_60', 'erosivity_US_adj_60']]
del result, final_result #clean not needed dataframes

# == # == # == # SAVE RESULTS # == # == == # == # 
if save_results:
    df_comparison.to_parquet(f"out/{station_num}_erosivity_comparison.parquet.gzip", compression="gzip")

# == # == # == # Plots # == # == == # == # 
# Define your x and y pairs
x = 'erosivity_US_5'
x_label = "Re_5min"
pairs = [
    ('RIST_EI30', 'Re_RIRST_5min'),
    ('erosivity_US_60', 'Re_60min'),
    ('erosivity_US_adj_60', 'Re_60min_corr')
]

# Set up the plot grid
axs_limit = 1400
fig, axs = plt.subplots(2, 2, figsize=(8.5, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Hide the last subplot to make space for the custom legend
axs[1, 1].axis('off')

# Create empty list to hold legend elements
legend_elements = []
scatter_plot = None  # To hold the scatter plot entry

# Loop through each pair and plot
colors = ['blue', 'green', 'red']  # Colors for the regression lines
for i, (y_col, label) in enumerate(pairs):
    ax = axs[i // 2, i % 2]  # Choose the appropriate subplot
    
    # Scatter plot (black points) for each subplot
    scatter_plot = ax.scatter(df_comparison[x], df_comparison[y_col], color='black', label="Erosivity in [MJ*mm/ha*hr]")
    
    # Linear Regression
    X = df_comparison[[x]]  # Independent variable (reshape to 2D)
    y = df_comparison[y_col]  # Dependent variable
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict and plot the regression line
    y_pred = model.predict(X)
    reg_line, = ax.plot(df_comparison[x], y_pred, color=colors[i], label=f"Regression: {label}")
    
    # Add the identity line (y = x) as a dashed black line
    identity_line, = ax.plot(np.linspace(0, axs_limit, 100), np.linspace(0, axs_limit, 100), 
                             color='black', linestyle='--', linewidth=0.8, label='Identity Line')
    
    # Calculate R²
    r2 = model.score(X, y)
    
    # Calculate RMSE between actual x and y (point-based)
    rmse = sqrt(mean_squared_error(df_comparison[x], df_comparison[y_col]))
    
    # Calculate Bias (mean of y/x)
    bias = np.mean(df_comparison[y_col] / df_comparison[x])
    
    # Add R², RMSE, and Bias to the plot
    ax.text(0.05, 0.95, f"R²: {r2:.2f}\nRMSE: {rmse:.2f}\nMBR: {bias:.2f}", 
            transform=ax.transAxes, 
            fontsize=10, 
            verticalalignment='top', 
            horizontalalignment='left', 
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    # Set axis limits
    ax.set_xlim(0, axs_limit)
    ax.set_ylim(0, axs_limit)
    
    # Labels (no title as requested)
    # Labels (use x_label and label from the loop)
    ax.set_xlabel(f'{x_label} [MJ*mm/ha*hr]')  # Dynamically set x-axis label
    ax.set_ylabel(f'{label} [MJ*mm/ha*hr]')  # Dynamically set y-axis label

    # Set x and y ticks with a step of 200 starting from 0
    ax.set_xticks(np.arange(0, axs_limit+1, 200))  # x-ticks from 0 to 1300 with step size 200
    ax.set_yticks(np.arange(0, axs_limit+1, 200))  # y-ticks from 0 to 1300 with step size 200

    # Add grid
    ax.grid(True)
    
    if identity_line.get_label() not in [elem[0].get_label() for elem in legend_elements]:
        legend_elements.append((identity_line, 'Identity Line'))

    # Add legend elements to list (avoid duplicates by checking if already added)
    if reg_line not in [elem[0] for elem in legend_elements]:
        legend_elements.append((reg_line, f"Regression: {label}"))

# Create a custom legend using all legend elements
# Only add the scatter plot once to avoid duplicates
fig.legend([scatter_plot] + [elem[0] for elem in legend_elements], 
           ['Erosivity Re in [MJ*mm/ha*hr]'] + [elem[1] for elem in legend_elements], 
           loc='center', 
           fontsize=14, 
           frameon=False, 
           title="Legend", 
           title_fontsize=16, 
           bbox_to_anchor=(0.7, 0.3))
fig.suptitle(f"{station_num} Erosivity Re in [MJ*mm/ha*hr] for events", fontsize=14, fontweight='bold', y=0.93)
if save_results:
    fig.savefig('fig/fig00_Re_comparison.jpeg', format='jpeg', dpi=300)
plt.show()




# Get lengths of the DataFrames
lengths = [
    ('RE 5 min from RIST', len(df_erosivity_5min_RIST)),
    ('Re 5min', len(df_erosivity_5)),
    ('Re 60min \n(German IMax30 threshold) \n5.79', len(df_erosivity_60))  
]

# Create the figure
fig, ax = plt.subplots(figsize=(6, 2))  # Adjust the size as needed

# Hide the axes (since we want to show only the table)
ax.axis('off')

# Create the table data (without including the column labels in the data)
table_data = [ [name, length] for name, length in lengths]

# Create the table in the figure
table = ax.table(cellText=table_data, loc='center', colLabels=['DataFrame', 'Length'], cellLoc='center', colColours=['#f5f5f5', '#f5f5f5'])

# Style the table (optional)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Adjust row height and column width (optional for better appearance)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        # Apply bold style to header row
        cell.set_fontsize(14)
        cell.set_text_props(weight='bold')
    cell.set_height(0.45)  # Increase row height
    cell.set_width(0.7)    # Increase column width

# Show the figure
if save_results:
    fig.savefig('fig/fig00_RE_datasets_lenght.jpeg', format='jpeg', dpi=300, bbox_inches='tight')
plt.show()