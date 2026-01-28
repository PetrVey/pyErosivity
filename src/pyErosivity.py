# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:25:32 2024

@author: Petr
"""
import pandas as pd
import numpy as np
from packaging.version import parse

def remove_incomplete_years(data_pr, name_col = 'value', nan_to_zero=True, tolerance=0.1):
    """
    Function that delete incomplete years in precipitation data.
    
    Parameters
    ----------
    data_pr : pd dataframe
        dataframe containing the hourly values of precipitation
    name_col : string
        name of column where variable values are stored 
    nan_to_zero: bool
        push nan to zero
    tolerance : float
        Max fraction of missing data in one year [-]
         
    Returns
    -------
    data_cleanded: pd dataframe 
       cleaned dataset.

    """
    # Step 1: get resolution of dataset (MUST BE SAME in whole dataset!!!)
    time_res = (data_pr.index[-1] - data_pr.index[-2]).total_seconds()/60
    # Step 2: Resample by year and count total and NaN values
    if parse(pd.__version__) > parse("2.2"):
        yearly_valid = data_pr.resample('YE').apply(lambda x: x.notna().sum())  # Count not NaNs per year
    else: 
        yearly_valid = data_pr.resample('Y').apply(lambda x: x.notna().sum())  # Count not NaNs per year
    # Step 3: Estimate expected lenght of yearly timeseries
    expected = pd.DataFrame(index = yearly_valid.index)
    expected["Total"] = 1440/time_res*365
    # Step 4: Calculate percentage of missing data per year by aligning the dimensions
    valid_percentage = (yearly_valid[name_col] / expected['Total'])       
    # Step 3: Filter out years where more than 10% of the values are NaN
    years_to_remove = valid_percentage[valid_percentage < 1-tolerance].index
    # Step 4: Remove data for those years from the original DataFrame
    data_cleanded = data_pr[~data_pr.index.year.isin(years_to_remove.year)]
    # Replace NaN values with 0 in the specific column
    if nan_to_zero:
        data_cleanded.loc[:, name_col] =  data_cleanded[name_col].fillna(0)
        
    time_resolution = time_res
    return data_cleanded, time_resolution


def get_events(data, dates, separation, min_rain, name_col='value', check_gaps=True):
    """
    
    Function that extracts precipitation events out of the entire data.
    
    Parameters
    ----------
    - data np.array: array containing the hourly values of precipitation.
    - separation (int): The number of hours used to define an independet ordianry event. Defult: 24 hours. this is saved in SMEV S class
                    Days with precipitation amounts above this threshold are considered events.
    - name_col (string): The name of the df column with precipitation values
    - check_gaps (bool): This also check for gaps in data and for unknown start/end events
    - min_rain : float
        minimum rainfall value, 
        reason --> Climate models has issue with too small float values (drizzles, eg. 0.0099mm/h)
               --> Another reason is that the that rain gauge tipping bucket has min value
    Returns
    -------
    - consecutive_values np.array: index of time of consecutive values defining the events.


    Examples
    --------
    """
    if isinstance(data,pd.DataFrame):
        # Find values above threshold
        above_threshold = data[data[name_col] > min_rain]
        # Find consecutive values above threshold separated by more than 24 observations
        consecutive_values = []
        temp = []
        for index, row in above_threshold.iterrows():
            if not temp:
                temp.append(index)
            else:
                if index - temp[-1] > pd.Timedelta(hours=separation):
                    if len(temp) >= 1:
                        consecutive_values.append(temp)
                    temp = []
                temp.append(index)
        if len(temp) >= 1:
            consecutive_values.append(temp)
            
    elif isinstance(data,np.ndarray):

        # Assuming data is your numpy array
        # Assuming name_col is the index for comparing threshold
        # Assuming threshold is the value above which you want to filter

        above_threshold_indices = np.where(data > min_rain)[0]

        # Find consecutive values above threshold separated by more than 24 observations
        consecutive_values = []
        temp = []
        for index in above_threshold_indices:
            if not temp:
                temp.append(index)
            else:
                #numpy delta is in nanoseconds, it  might be better to do dates[index] - dates[temp[-1]]).item() / np.timedelta64(1, 'm')
                if (dates[index] - dates[temp[-1]]).item() > (separation * 3.6e+12):  # Assuming 24 is the number of hours, nanoseconds * 3.6e+12 = hours
                    if len(temp) >= 1:
                        consecutive_values.append(dates[temp])
                    temp = []
                temp.append(index)
        if len(temp) >= 1:
            consecutive_values.append(dates[temp])
    
    if check_gaps == True:
        #remove event that starts before dataset starts in regard of separation time
        if (consecutive_values[0][0] - dates[0]).item() < (separation * 3.6e+12): #this numpy dt, so still in nanoseconds
            consecutive_values.pop(0)
        else:
            pass
        
        #remove event that ends before dataset ends in regard of separation time
        if (dates[-1] - consecutive_values[-1][-1]).item() < (separation * 3.6e+12): #this numpy dt, so still in nanoseconds
            consecutive_values.pop()
        else:
            pass
        
        #Locate OE that ends before gaps in data starts.
        # Calculate the differences between consecutive elements
        time_diffs = np.diff(dates)
        #difference of first element is time resolution
        time_res = time_diffs[0]
        # Identify gaps (where the difference is greater than 1 hour)
        gap_indices_end = np.where(time_diffs > np.timedelta64(int(separation * 3.6e+12), 'ns'))[0]
        # extend by another index in gap cause we need to check if there is OE there too
        gap_indices_start = ( gap_indices_end  + 1)
       
        match_info = []
        for gap_idx in gap_indices_end:
            end_date = dates[gap_idx]
            start_date = end_date - np.timedelta64(int(separation * 3.6e+12), 'ns')
            # Creating an array from start_date to end_date in hourly intervals
            temp_date_array = np.arange(start_date, end_date, time_res)
            
            # Checking for matching indices in consecutive_values
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                if match_indices.size > 0:
                    
                    match_info.append(i)
         
        for gap_idx in gap_indices_start:
            start_date = dates[gap_idx]
            end_date = start_date + np.timedelta64(int(separation * 3.6e+12), 'ns')
            # Creating an array from start_date to end_date in hourly intervals
            temp_date_array = np.arange(start_date, end_date, time_res)
            
            # Checking for matching indices in consecutive_values
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                if match_indices.size > 0:
                    
                    match_info.append(i)
                    
        for del_index in sorted( match_info, reverse=True):
            del consecutive_values[del_index]
                
    return consecutive_values


def split_event_by_6h_threshold_dates(event_dates, dates, data, window_steps, date_to_idx, thresh=1.27):
    """
    Split a preliminary storm into sub-events using 6-hour accumulation.
    event_dates : array of numpy.datetime64
    returns     : list of arrays of numpy.datetime64
    """

    # Map dates → indices
    event_idx = np.array([date_to_idx[d] for d in event_dates])

    # Rolling sum on the **full data**, then select event indices
    full_roll = np.convolve(data, np.ones(window_steps), mode="same")
    roll = full_roll[event_idx]  # only values corresponding to this event

    wet = roll > thresh

    splits = []
    temp = []

    for d, flag in zip(event_dates, wet):
        if flag:
            temp.append(d)
        else:
            if temp:
                splits.append(np.array(temp))
                temp = []

    if temp:
        splits.append(np.array(temp))

    return splits




def get_events_Renard_RUSLE(data, dates, separation, time_resolution, check_gaps=True, ):
    """
    
    Function that extracts precipitation events out of the entire data.
    A rainfall accumulation of less than 1.27 mm during a period of 6 h splits a longer storm period into two storms. 
    
    Please be aware that this function actually deletes drizzles as it marks end of storm if rolling mean accumulated precipitation in 6h is less than 1.27.
    example on 14h storm:
    
hour |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  | 13  | 14  |
      |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
prec  | 0.4 | 1.2 |  5  | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 10  | 0.2 |
6h acc| 7.2 |  7  |  6  | 1.2 | 1.2 | 1.2 | 1.2 | 11  | 11  |10.8 |10.6 |10.4 |10.2 | 0.2 |
      |S#1  |     |END#1|     |     |     |     |S#2  |     |     |     |     |     |END#2|
      |     |     |Split|     |     |     |     |     |     |     |     |     |     |     |
      |     |     |Point|     |     |     |     |     |     |     |     |     |     |     |    
    """
    min_rain = 0
    above_threshold_indices = np.where(data > min_rain)[0]
    

    # Find consecutive values above threshold separated by more than 24 observations
    consecutive_values_temp = []
    temp = []
    for index in above_threshold_indices:
        if not temp:
            temp.append(index)
        else:
            #numpy delta is in nanoseconds, it  might be better to do dates[index] - dates[temp[-1]]).item() / np.timedelta64(1, 'm')
            if (dates[index] - dates[temp[-1]]).item() > (separation * 3.6e+12):  # Assuming 24 is the number of hours, nanoseconds * 3.6e+12 = hours
                if len(temp) >= 1:
                    consecutive_values_temp.append(dates[temp])
                temp = []
            temp.append(index)
    if len(temp) >= 1:
        consecutive_values_temp.append(dates[temp])
        
    date_to_idx = {d: i for i, d in enumerate(dates)}
    
    dt_hours = time_resolution / 60
    window_steps = int(6 / dt_hours)
        
    consecutive_values = []
    
    for event_dates in consecutive_values_temp:
        sub_events = split_event_by_6h_threshold_dates(
            event_dates=event_dates,
            dates=dates,
            data=data,
            window_steps=window_steps,
            date_to_idx=date_to_idx,
            thresh=1.27
        )
        consecutive_values.extend(sub_events)

    if check_gaps == True:
        #remove event that starts before dataset starts in regard of separation time
        if (consecutive_values[0][0] - dates[0]).item() < (separation * 3.6e+12): #this numpy dt, so still in nanoseconds
            consecutive_values.pop(0)
        else:
            pass
        
        #remove event that ends before dataset ends in regard of separation time
        if (dates[-1] - consecutive_values[-1][-1]).item() < (separation * 3.6e+12): #this numpy dt, so still in nanoseconds
            consecutive_values.pop()
        else:
            pass
        
        #Locate OE that ends before gaps in data starts.
        # Calculate the differences between consecutive elements
        time_diffs = np.diff(dates)
        #difference of first element is time resolution
        time_res = time_diffs[0]
        # Identify gaps (where the difference is greater than 1 hour)
        gap_indices_end = np.where(time_diffs > np.timedelta64(int(separation * 3.6e+12), 'ns'))[0]
        # extend by another index in gap cause we need to check if there is OE there too
        gap_indices_start = ( gap_indices_end  + 1)
       
        match_info = []
        for gap_idx in gap_indices_end:
            end_date = dates[gap_idx]
            start_date = end_date - np.timedelta64(int(separation * 3.6e+12), 'ns')
            # Creating an array from start_date to end_date in hourly intervals
            temp_date_array = np.arange(start_date, end_date, time_res)
            
            # Checking for matching indices in consecutive_values
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                if match_indices.size > 0:
                    
                    match_info.append(i)
         
        for gap_idx in gap_indices_start:
            start_date = dates[gap_idx]
            end_date = start_date + np.timedelta64(int(separation * 3.6e+12), 'ns')
            # Creating an array from start_date to end_date in hourly intervals
            temp_date_array = np.arange(start_date, end_date, time_res)
            
            # Checking for matching indices in consecutive_values
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                if match_indices.size > 0:
                    
                    match_info.append(i)
                    
        for del_index in sorted( match_info, reverse=True):
            del consecutive_values[del_index]
                
    return consecutive_values

def remove_short(list_events:list, time_resolution=None, min_ev_dur=None):
     """
     
     Function that removes events events too short.
     
     Parameters
     ----------
     - list_events list: list of indices of events events as returned by `get_events_events`.
     - time_resolution: Used to calculate lenght of storm [mins]
     - min_ev_dur : int
         Minimum event duration [min]
     Returns
     -------
     - arr_vals : boolean array, 
     - arr_dates : dates of OE in TO, FROM format
     - n_events_per_year: count of OE in each years

     Examples
     --------
     """
     if time_resolution==None or min_ev_dur==None:
         print("Warning !!!! Warning !!!! Warning !!!! Warning !!!! Warning !!!!")
         print("Warning !!!! Warning !!!! Warning !!!! Warning !!!! Warning !!!!")
         print("time_resolution or min_ev_duration not provided")
         arr_vals,arr_dates,n_events_per_year = np.nan,np.nan,np.nan
     else:
         if isinstance(list_events[0][0],pd.Timestamp):
             # event is multiplied by its lenght to get duration and compared with min_duration setup
             ll_short=[True if ev[-1]-ev[0] + pd.Timedelta(minutes=time_resolution) >= pd.Timedelta(minutes=min_ev_dur) else False for ev in list_events]
             ll_dates=[(ev[-1].strftime("%Y-%m-%d %H:%M:%S"),ev[0].strftime("%Y-%m-%d %H:%M:%S")) if ev[-1]-ev[0] + pd.Timedelta(minutes=time_resolution) >= pd.Timedelta(minutes=min_ev_dur)
                       else (np.nan,np.nan) for ev in list_events]
             arr_vals=np.array(ll_short)[ll_short]
             arr_dates=np.array(ll_dates)[ll_short]
             filtered_list = [x for x, keep in zip(list_events, ll_short) if keep]
             list_year=pd.DataFrame([filtered_list[_][0].year for _ in range(len(filtered_list))],columns=['year'])
             n_events_per_year=list_year.reset_index().groupby(["year"]).count()
             # n_events=n_events_per_year.mean().values.item()
         elif isinstance(list_events[0][0],np.datetime64):
             ll_short=[True if (ev[-1]-ev[0]).astype('timedelta64[m]')+ np.timedelta64(int(time_resolution),'m') >= pd.Timedelta(minutes=min_ev_dur) else False for ev in list_events]
             ll_dates=[(ev[-1],ev[0]) if (ev[-1]-ev[0]).astype('timedelta64[m]') + np.timedelta64(int(time_resolution),'m') >= pd.Timedelta(minutes=min_ev_dur) else (np.nan,np.nan) for ev in list_events]
             arr_vals=np.array(ll_short)[ll_short]
             arr_dates=np.array(ll_dates)[ll_short]
  
             filtered_list = [x for x, keep in zip(list_events, ll_short) if keep]
             list_year=pd.DataFrame([filtered_list[_][0].astype('datetime64[Y]').item().year for _ in range(len(filtered_list))],columns=['year'])
             n_events_per_year=list_year.reset_index().groupby(["year"]).count()
             # n_events=n_events_per_year.mean().values.item()

     return arr_vals,arr_dates,n_events_per_year


def get_events_values(data, dates, arr_dates_oe, durations=[], time_resolution=None):
    """
    Parameters
    ----------
    data : np array
        data of full dataset 
    dates : np array
        time of full dataset 
    arr_dates_oe : TYPE
        end and start of event, this is output from remove_short function.
    durations: List
        List of durations in minutes eg [30,60]
    time_resolution = integer or float
        time resolution in dataset in minutes
    Returns
    -------
    dict_events : dict of pandas 
        events per duration.
        dict_events = {"10" : pd.DataFrame(columns=['year', 'event_start', 'event_end', 'event_peak', 'prec_depth',
                                                      'intensity_per_hour', 'prec_accum', 'E_kin', 'erosivity'],}
        Here explanation of each column:
            year --> year of event
           
            event_start  --> event start [time]
            event_end  --> event end [time]
            event_peak --> time of peak which is given per accumulate duration, it is not peak of storm [time]
            prec_depth --> Maximum accumulated depth for given duration [mm]
            intensity_per_hour -->  Maximum precipitation intensity for given duration eg Imax30 [mm/h]
            prec_accum  -->  Total accumulated depth for event [mm]
            E_kin --> Ekin for event (kinetic energy of precipitation event) [kJ m^−2]
            erosivity_EU --> Erosivity of event in [kJ/m^2 * mm/h = N/h], In EU erosivity factors are given in this unit
                            NOTE!!!
                            R factors (Erosivity) are often given in the unit MJ*mm/ha*hr (in US)
                            To convert rainfall erosivity as given here in N/h to MJ*mm/ha*hr, it has to be multiplied by a factor of 10.
            erosivity_US --> erosivity of event in MJ*mm/ha*hr
        
    """
    dict_events = {}
    
    if time_resolution == None or not durations:
        print("Warning !!!! Warning !!!! Warning !!!! Warning !!!! Warning !!!!")
        print("Warning !!!! Warning !!!! Warning !!!! Warning !!!! Warning !!!!")
        print("time resolution or durations not provided")
    else:
        for d in range(len(durations)):
            arr_conv = np.convolve(data, np.ones(int(durations[d]/time_resolution),dtype=int),'same')
        
            # Convert time index to numpy array
            time_index = dates.reshape(-1)
        
            # Use numpy indexing to get the max values efficiently
            ll_vals = [] # Maximum accumulated depth for given duration 
            ll_intensities = [] # Maximum precipitation intensity for given duration eg Imax30 mm/h
            ll_accums = [] # Total accumulated depth for event 
            ll_dates = [] # Time of peak for accumulate depth
            ll_starts = [] #event start
            ll_ends = [] #event end
            ll_Ekins= [] # Ekin for event (kinetic energu of precipitation event)
            ll_Res = [] # Erosivity of event
            
            for i in range(arr_dates_oe.shape[0]):
                start_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 1])
                   
                end_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 0])
                    
                # Check if start and end times are the same
                if start_time_idx == end_time_idx:
                    ll_val = arr_conv[start_time_idx]
                    ll_date = time_index[start_time_idx]
                    ll_start = time_index[start_time_idx]
                    ll_end = time_index[end_time_idx]
                    
                    ll_intesnity =  ll_val * 60 / durations[d] 
                    
                    # Get accumulated precpitaiton in event    
                    ll_accum  = np.sum(data[start_time_idx])
                    
                    # Peak time
                    ll_date = time_index[start_time_idx]
                    
                    # Depth at each step (increment)
                    ll_depth = data[ start_time_idx]
                    # Calculcate ekin_i
                    ll_hourly_intensities = data[ start_time_idx] * 60 / time_resolution
                    # Vectorize the function for calculcation ekin at increment t
                    E_kin_i_vectorized = np.vectorize(E_kin_i)
                    # Apply the function to get Ekin at increment t
                    ll_Ekin_i = E_kin_i_vectorized(ll_hourly_intensities)
                    ll_Ekin = np.sum(ll_Ekin_i * ll_depth)
                    
                    # the erosivity of rain event Re (N h−1)
                    ll_Re = ll_Ekin * ll_intesnity
                    
                else:
                    # do convolve inside of the event
                    arr_conv2 = np.convolve(data[start_time_idx : end_time_idx + 1],
                                           np.ones(int(durations[d] / time_resolution), dtype=int),
                                           "same",
                                       )
                    
                    
                    # the +1 in end_time_index is because then we search by index but we want to includde last as well,
                    # without, it slices eg. end index is 10, without +1 it slices 0 to 9 instead of 0 to 10 (stops 1 before)    
                    # get index of ll_val within the sliced array
                    #old version ll_idx_in_slice = np.nanargmax(arr_conv[start_time_idx:end_time_idx+1]) 
                    ll_idx_in_slice = np.nanargmax(arr_conv2)
                    
                    # Adjust the index to refer to the original arr_conv
                    ll_idx_in_arr_conv = start_time_idx + ll_idx_in_slice
                    
                    # Get max value -> peak
                    #old version ll_val = arr_conv[ll_idx_in_arr_conv] 
                    ll_val = arr_conv2[ll_idx_in_slice]
                    
                    # Get intensity in mm/h
                    ll_intesnity =  ll_val * 60 / durations[d] 
                    
                    # Get accumulated precpitaiton in event    
                    ll_accum  = np.sum(data[start_time_idx:end_time_idx+1])
                    
                    # Peak time
                    ll_date = time_index[ll_idx_in_arr_conv]
                    
                    # Start and end 
                    ll_start = time_index[start_time_idx]
                    ll_end = time_index[end_time_idx]
                    
                    # Depth at each step (increment)
                    ll_depth = data[start_time_idx:end_time_idx+1]
                    # Calculcate ekin_i
                    ll_hourly_intensities = data[start_time_idx:end_time_idx+1] * 60 / time_resolution
                    # Vectorize the function for calculcation ekin at increment t
                    E_kin_i_vectorized = np.vectorize(E_kin_i)
                    # Apply the function to get Ekin at increment t
                    ll_Ekin_i = E_kin_i_vectorized(ll_hourly_intensities)
                    ll_Ekin = np.sum(ll_Ekin_i * ll_depth)
                    
                    # the erosivity of rain event Re (N h−1)
                    ll_Re = ll_Ekin * ll_intesnity
                    
                ll_vals.append(ll_val)
                ll_intensities.append(ll_intesnity)
                ll_accums.append(ll_accum)
                ll_dates.append(ll_date)
                ll_starts.append(ll_start)
                ll_ends.append(ll_end)
                ll_Ekins.append(ll_Ekin)
                ll_Res.append(ll_Re)
            #years  of events events
            ll_yrs=[arr_dates_oe[_,0].astype('datetime64[Y]').item().year for _ in range(arr_dates_oe.shape[0])]
            
            df_oe = pd.DataFrame({'year':ll_yrs,
                                  'event_start': ll_starts,
                                  'event_end': ll_ends,
                                  'event_peak':ll_dates,
                                  'prec_depth':ll_vals,
                                  'intensity_per_hour': ll_intensities,
                                  'prec_accum': ll_accums,
                                  'E_kin':  ll_Ekins,
                                  'erosivity_EU': ll_Res,
                                  'erosivity_US': [x*10 for x in ll_Res]})
            dict_events.update({f"{durations[d]}":df_oe})
  
    return dict_events

# Define the function to calculate kinetic energy per time interval
def E_kin_i(intensity):
    # based on DIN 19708
    # which is based on 
    # Rogler, H. and Schwertmann, U.: 
    # Erosivität der Niederschläge und Isoerodentkarte Bayerns, J. Rural Engi. Developm., 22, 99–112, 1981. 
    
    #The upper limit is set because raindrops do not continue to grow significantly indefinitely; 
    #their size and fall velocity approach practical maxima” (Van Dijk et al., 2002).
    
    if intensity < 0.05:
        return 0
    elif intensity >= 76.2:
        return 28.33 * 10**-3
    else:
        return (11.89 + 8.73 * np.log10(intensity)) * 10**-3
    
def E_kin_i_BrFr(intensity): 
    #(Brown and Foster, 1987)
    # not in use, buyt if it's used, it gives E_kin in US units ha instead of EU calculation,
    # so it will messed other things, rather dont use it, maybe just for testing.
    return 0.29*(1-0.72*np.exp(-0.05*intensity))
    
def get_only_erosivity_events(df, accum_threshold=12.7, intensity_threshold=12.7, use_both_thresholds=True):
    """
    Wischmeier 1959, 1979
    Also used in Rogler and Schwertmann 1981 and DIN 19708:2017-08

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    accum_threshold : TYPE, optional
        DESCRIPTION. The default is 12.7.
    intensity_threshold : TYPE, optional
        DESCRIPTION. The default is 12.7.
    use_both_thresholds : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    filtered_df : TYPE
        DESCRIPTION.

    """
    if use_both_thresholds:
        filtered_df = df[(df['intensity_per_hour'] >= intensity_threshold) | (df['prec_accum'] >= accum_threshold)]
    else:
        filtered_df = df[(df['intensity_per_hour'] >= intensity_threshold)]
        
    filtered_df = filtered_df.reset_index(drop=True)    
    return filtered_df


def get_only_erosivity_events_Renard(df, accum_threshold=12.7, depth_threshold=12.7):
    """
    This function is consistent with Renard 1997 (RUSLE)
    
   criteria for the identification of an erosive event are given:
        (i) the cumulative rainfall of an event is greater than 12.7 mm, or 
        (ii) the event has at least one peak that is greater than 6.35 mm during a period of 15 min 
        (or 12.7 mm during a period of 30 min). 
    
    Keep in mind that this is designed for the high temporal resolution data. where we can estimate correctly Imax15 and Imax30.
    If we have hourly data, this treshold must be adjusted. 
    
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    accum_threshold : TYPE, optional
        DESCRIPTION. The default is 12.7.
    intensity_threshold : TYPE, optional
        DESCRIPTION. The default is 12.7.
    use_both_thresholds : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    filtered_df : TYPE
        DESCRIPTION.

    """

    filtered_df = df[(df['prec_depth'] >= depth_threshold) | (df['prec_accum'] >= accum_threshold)]

    filtered_df = filtered_df.reset_index(drop=True)    
    return filtered_df

def boostrapping_erosivity_60min ( df_erosivity, 
                                  niter=1000,
                                  M=None):
    """
    Warning, this function is for boostrapping erosivity out of 60min dataset
    based on event peak year.

    
    Parameters
    ----------
    df_erosivity : pandas
        df_erosivity is pandas which is given 
        by fuctnion "get_only_erosivity_events" and 
        after calculcting the adjusted erosivity
        for 60 min data
        
    niter : int, optional
        N of boostrap samples (N of boostrap iteration). The default is 1000.
        
    M : int, optional
        Number of boostrap sample lenght in years.
        eg. 10 gives 10 years data lenght etc.

    Returns
    -------
    f_bootstrap_summary

    """
    # Ensure 'event_peak' is datetime
    df_erosivity['event_peak'] = pd.to_datetime(df_erosivity['event_peak'])

    # extract years out of population
    blocks = np.unique(df_erosivity['event_peak'].dt.year)
    # M 
    if M == None:
        M = len(blocks)
    else:
        pass
    
    # Create bootstrap samples as random combination of years
    randy = np.random.choice(blocks, size=(niter, M), replace=True)
    
    # Precompute yearly aggregates once
    yearly_agg = df_erosivity.groupby('year').agg(
        N_events=('event_peak', 'count'),
        mean_intensity_per_hour=('intensity_per_hour', 'mean'),
        mean_prec_accum=('prec_accum', 'mean'),
        sum_erosivity_US_adj=('erosivity_US_adj', 'sum')
    ).reset_index()
    
    # Then bootstrap sample years from yearly_agg, sum/mean accordingly
    bootstrap_summaries = []
    
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
    
        overall_means = sample_df[['N_events', 'mean_intensity_per_hour', 'mean_prec_accum', 'sum_erosivity_US_adj']].mean()
    
        overall_means.index = [
            'mean_annual_events',
            'mean_annual_Imax',
            'mean_rain_depth',
            'average_annual_erosivity'
        ]
    
        overall_means['sample'] = f'sample_{i}'
        bootstrap_summaries.append(overall_means)
    
    df_bootstrap_summary = pd.DataFrame(bootstrap_summaries).set_index('sample')
    
    return df_bootstrap_summary

    
def boostrapping_erosivity_CPM_60min ( df_erosivity, 
                                      niter=1000,
                                      M=None,
                                      randy=None):
    """
    Warning, this function is for boostrapping erosivity out of 60min dataset
    based on event peak year.

    
    Parameters
    ----------
    df_erosivity : pandas
        df_erosivity is pandas which is given 
        by fuctnion "get_only_erosivity_events" and 
        after calculcting the adjusted erosivity
        for 60 min data
        
    niter : int, optional
        N of boostrap samples (N of boostrap iteration). The default is 1000.
        
    M : int, optional
        Number of boostrap sample lenght in years.
        eg. 10 gives 10 years data lenght etc.
        
    randy: numpy array
        predefined random requance of bootstrap samples
        each row must have set of N years, lenght of sample isn't limited.
        If randy is given, the niter and M is ignored.

    Returns
    -------
    f_bootstrap_summary

    """
    # Ensure 'event_peak' is datetime
    df_erosivity['event_peak'] = pd.to_datetime(df_erosivity['event_peak'])
    
    if randy is None:
        # extract years out of population
        blocks = np.unique(df_erosivity['event_peak'].dt.year)
        # M 
        if M == None:
            M = len(blocks)
        else:
            pass
        
        # Create bootstrap samples as random combination of years
        randy = np.random.choice(blocks, size=(niter, M), replace=True)
    else:
        #randy bust be integers 
        if not randy.dtype == np.int32:
            randy = randy.astype(np.int32)
            
        unique_randy_vals = np.unique(randy)  # e.g., [1, 2, ..., N]
        n_randy_vals = len(unique_randy_vals)
        
        blocks = np.sort(df_erosivity['event_peak'].dt.year.unique())
        
        if len(blocks) != n_randy_vals:
            raise ValueError(f"Mismatch: randy has {n_randy_vals} unique values, but df_erosivity has {len(blocks)} unique years.")


        # Check for clean 1-based consecutive values
        # meaning is that randy should have consectuive values as unique
        # like 1 to 10, in such case, we will do fast indexing
        # if randy has years 1 to 5 and then 10 to 15, we will do more robust mapping
        is_1_based_consecutive = (
            np.array_equal(unique_randy_vals, np.arange(1, n_randy_vals + 1))
        )
        
        if is_1_based_consecutive:
            # Use fast indexing
            randy = blocks[randy - 1]
        else:
            # Use robust mapping
            mapping = dict(zip(unique_randy_vals, blocks))
            randy = np.vectorize(mapping.get)(randy)
    
    # Precompute yearly aggregates once
    yearly_agg = df_erosivity.groupby('year').agg(
        N_events=('event_peak', 'count'),
        mean_intensity_per_hour=('intensity_per_hour', 'mean'),
        mean_prec_accum=('prec_accum', 'mean'),
        sum_erosivity_US_adj=('erosivity_US_adj', 'sum')
    ).reset_index()
    
    # Then bootstrap sample years from yearly_agg, sum/mean accordingly
    bootstrap_summaries = []
    
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
    
        overall_means = sample_df[['N_events', 'mean_intensity_per_hour', 'mean_prec_accum', 'sum_erosivity_US_adj']].mean()
    
        overall_means.index = [
            'mean_annual_events',
            'mean_annual_Imax',
            'mean_rain_depth',
            'average_annual_erosivity'
        ]
    
        overall_means['sample'] = f'sample_{i}'
        bootstrap_summaries.append(overall_means)
    
    df_bootstrap_summary = pd.DataFrame(bootstrap_summaries).set_index('sample')
    
    return df_bootstrap_summary