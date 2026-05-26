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
    Remove years with too many missing values from a precipitation time series.

    Parameters
    ----------
    data_pr : pd.DataFrame
        DataFrame with a DatetimeIndex containing precipitation values.
    name_col : str, optional
        Name of the column with precipitation values. Default 'value'.
    nan_to_zero : bool, optional
        Replace remaining NaNs with 0 after filtering. Default True.
    tolerance : float, optional
        Maximum allowed fraction of missing values per year [0–1]. Default 0.1 (10%).

    Returns
    -------
    data_cleaned : pd.DataFrame
        Filtered dataset with incomplete years removed.
    time_resolution : float
        Detected time step of the dataset [minutes].
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
    Extract precipitation events from a continuous time series.

    Two rain periods are considered the same event if the gap between them is
    shorter than `separation` hours. Events touching data gaps or dataset
    boundaries within `separation` hours are discarded (check_gaps=True).

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Precipitation values.
    dates : np.ndarray
        Timestamps corresponding to data (used when data is np.ndarray).
    separation : int or float
        Minimum dry gap [hours] required to split two events.
    min_rain : float
        Minimum precipitation value to be considered non-zero. Filters out
        gauge noise and climate-model drizzle artefacts (e.g. 0.001 mm/h).
    name_col : str, optional
        Column name for precipitation values when data is a DataFrame. Default 'value'.
    check_gaps : bool, optional
        If True, remove events whose start/end falls within `separation` hours
        of a data gap or the dataset boundary. Default True.

    Returns
    -------
    consecutive_values : list of arrays
        Each element is an array of timestamps (np.datetime64) or pd.Timestamps
        belonging to one event.
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
        dates = dates.astype("datetime64[ns]")

        if min_rain == 0:
            above_threshold_indices = np.where(data > min_rain)[0]
        else:
            above_threshold_indices = np.where(data >= min_rain)[0]

        if len(above_threshold_indices) == 0:
            return []

        # Vectorized event grouping: find gaps between consecutive wet steps,
        # then split at those gaps — O(n) vs the old O(n) Python loop but ~10x faster
        above_dates = dates[above_threshold_indices]
        time_diffs_above = np.diff(above_dates).astype(np.int64)
        separation_ns = int(separation * 3.6e12)
        split_points = np.where(time_diffs_above > separation_ns)[0] + 1
        index_groups = np.split(above_threshold_indices, split_points)
        consecutive_values = [dates[group] for group in index_groups]
    
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
    Split a storm into sub-events based on the Renard et al. (1997) RUSLE 6-hour rule.

    Per RUSLE (Renard 1997): if accumulated rainfall in any 6-hour period within a storm
    is less than 1.27 mm (0.05 in), that period is treated as a dry break and the storm
    is split into two separate erosive events at that point.

    Parameters
    ----------
    event_dates : np.ndarray of np.datetime64
        Timestamps of the preliminary storm (all wet steps).
    dates : np.ndarray of np.datetime64
        Full dataset timestamps (used to compute the rolling sum over the whole series).
    data : np.ndarray
        Full precipitation array aligned with dates.
    window_steps : int
        Number of time steps in a 6-hour window (= 6h / time_resolution).
    date_to_idx : dict
        Mapping from np.datetime64 timestamp to integer index in dates/data.
    thresh : float, optional
        Minimum 6-hour accumulation [mm] to keep a period as wet. Default 1.27 mm.

    Returns
    -------
    splits : list of np.ndarray
        Sub-event arrays of np.datetime64; each array is one continuous erosive period.
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
    Extract precipitation events using the Renard et al. (1997) RUSLE storm-splitting rule.

    Storms are first identified as continuous rain periods (any gap > separation hours ends an event).
    Each preliminary storm is then split further: if accumulated rainfall in any 6-hour window
    drops below 1.27 mm (0.05 in), that gap defines a boundary between two separate erosive events.
    This also removes isolated drizzle steps that fall below the 6-hour accumulation threshold.

    Example — a 14-hour storm split into two sub-events:


    
    hour |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10  | 11  | 12  | 13  | 14  |
         |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
    prec | 0.4 | 1.2 |  5  | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 10  | 0.2 |
    6h   | 7.2 |  7  |  6  | 1.2 | 1.2 | 1.2 | 1.2 | 11  | 11  |10.8 |10.6 |10.4 |10.2 | 0.2 |
         |S#1  |     |END#1|     |     |     |     |S#2  |     |     |     |     |     |END#2|

    Hours 4–7 have 6h accumulation = 1.2 mm < 1.27 mm → split point between event #1 and #2.

    Parameters
    ----------
    data : np.ndarray
        Full precipitation array.
    dates : np.ndarray of np.datetime64
        Timestamps aligned with data.
    separation : int or float
        Minimum dry gap [hours] to end a preliminary storm (passed to the initial event finder).
    time_resolution : int or float
        Data time step [minutes]. Used to compute the 6-hour window size.
    check_gaps : bool, optional
        Remove events touching data gaps or dataset boundaries. Default True.

    Returns
    -------
    consecutive_values : list of np.ndarray
        Each element is an array of np.datetime64 timestamps for one erosive event.
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
    Remove precipitation events shorter than a minimum duration.

    Parameters
    ----------
    list_events : list of arrays
        Output of get_events or get_events_Renard_RUSLE — each element is an array
        of timestamps for one event.
    time_resolution : int or float
        Data time step [minutes]. Required to compute event duration correctly
        (duration = last_step - first_step + time_resolution).
    min_ev_dur : int or float
        Minimum event duration [minutes]. Events shorter than this are removed.

    Returns
    -------
    arr_vals : np.ndarray
        Boolean array (all True) for events that passed the filter.
    arr_dates : np.ndarray, shape (N, 2)
        Array of (end, start) timestamp pairs for each retained event.
    n_events_per_year : pd.DataFrame
        Count of retained events per calendar year.
    """
    if time_resolution is None or min_ev_dur is None:
        raise ValueError("time_resolution and min_ev_dur must both be provided.")

    # Normalise to np.datetime64 so there is one unified code path
    if isinstance(list_events[0][0], pd.Timestamp):
        list_events = [
            np.array([t.to_datetime64() for t in ev])
            for ev in list_events
        ]

    min_duration = np.timedelta64(int(min_ev_dur), "m")
    time_res     = np.timedelta64(int(time_resolution), "m")

    ll_short = [
        (ev[-1] - ev[0]).astype("timedelta64[m]") + time_res >= min_duration
        for ev in list_events
    ]

    ll_dates = [
        (ev[-1], ev[0]) if keep else (np.nan, np.nan)
        for ev, keep in zip(list_events, ll_short)
    ]

    arr_vals  = np.array(ll_short)[ll_short]
    arr_dates = np.array(ll_dates)[ll_short]

    filtered_list = [ev for ev, keep in zip(list_events, ll_short) if keep]

    list_year = pd.DataFrame(
        [ev[0].astype("datetime64[Y]").item().year for ev in filtered_list],
        columns=["year"],
    )
    n_events_per_year = list_year.reset_index().groupby(["year"]).count()

    return arr_vals, arr_dates, n_events_per_year


def get_events_values(data, dates, arr_dates_oe, durations=[], time_resolution=None):
    """
    Compute erosivity-relevant metrics for each event and each accumulation duration.

    For each event and each duration in `durations`, a sliding-window convolution finds
    the peak accumulated depth (prec_depth) and converts it to intensity (intensity_per_hour).
    Kinetic energy (E_kin) and erosivity (EI30) are computed over the full event.

    Parameters
    ----------
    data : np.ndarray
        Full precipitation time series [mm per time step].
    dates : np.ndarray of np.datetime64
        Timestamps aligned with data.
    arr_dates_oe : np.ndarray, shape (N, 2)
        Event (end, start) pairs as returned by remove_short.
    durations : list of int
        Accumulation window lengths [minutes]. Must be [30] or [60].
        Use [30] for IMax30 (standard, any resolution <= 30 min).
        Use [60] for IMax60 = IMax30 at 60-min resolution
        (Williams & Sheridan 1991). Any other value raises ValueError.
    time_resolution : int or float
        Data time step [minutes].

    Returns
    -------
    dict_events : dict of pd.DataFrame
        Keys are duration strings (e.g. '30'). Each DataFrame has columns:
            year               : calendar year of event peak
            event_start        : event start timestamp
            event_end          : event end timestamp
            event_peak         : timestamp of peak accumulation window
            prec_depth         : maximum accumulated depth over the window [mm]
            intensity_per_hour : peak window intensity [mm/h]  (= prec_depth * 60 / duration)
            prec_accum         : total accumulated depth over the whole event [mm]
            E_kin              : kinetic energy of the event [kJ m-2]
            erosivity_EU       : event erosivity E_kin x IMax  [kJ m-2 mm h-1 = N h-1]
            erosivity_US       : same in US units [MJ mm ha-1 h-1]  (= erosivity_EU x 10)
    """
    if time_resolution is None or not durations:
        raise ValueError(
            "time_resolution and durations must both be provided."
        )
    _allowed = {30, 60}
    _invalid = set(durations) - _allowed
    if _invalid:
        raise ValueError(
            f"Invalid durations: {sorted(_invalid)}. "
            f"Only [30] or [60] are supported. "
        )

    dict_events = {}
    time_index = dates.reshape(-1)
    n_events = arr_dates_oe.shape[0]

    # Pre-compute start/end indices for all events at once (vectorized)
    oe_end   = arr_dates_oe[:, 0].astype("datetime64[ns]")
    oe_start = arr_dates_oe[:, 1].astype("datetime64[ns]")
    start_indices = np.searchsorted(time_index, oe_start)
    end_indices   = np.searchsorted(time_index, oe_end)

    # Year array is the same for every duration -- compute once
    ll_yrs = [oe_end[i].astype("datetime64[Y]").item().year for i in range(n_events)]

    # Integer data avoids float accumulation errors in convolution sums
    data_int = np.round(data * 10000).astype(np.int64)

    E_kin_i_vectorized = np.vectorize(E_kin_i)

    for d in range(len(durations)):
        window_size = int(durations[d] / time_resolution)
        ones_kernel = np.ones(window_size, dtype=np.int64)

        ll_vals        = []
        ll_intensities = []
        ll_accums      = []
        ll_dates       = []
        ll_starts      = []
        ll_ends        = []
        ll_Ekins       = []
        ll_Res         = []

        for i in range(n_events):
            si = start_indices[i]
            ei = end_indices[i]

            if si == ei:
                ll_val    = data[si]
                ll_date   = time_index[si]
                ll_start  = time_index[si]
                ll_end    = time_index[ei]
                ll_depth  = data[si]
                ll_accum  = float(np.sum(data[si]))
                ll_hourly_intensities = data[si] * 60 / time_resolution
            else:
                # +1 on end so slice includes the last step
                # Integer convolution avoids float accumulation errors
                arr_conv_int = np.convolve(data_int[si:ei + 1], ones_kernel, "same")
                ll_idx   = np.nanargmax(arr_conv_int)
                ll_val   = arr_conv_int[ll_idx] / 10000.0
                ll_date  = time_index[si + ll_idx]
                ll_start = time_index[si]
                ll_end   = time_index[ei]
                ll_depth = data[si:ei + 1]
                ll_accum = float(np.sum(data[si:ei + 1]))
                ll_hourly_intensities = data[si:ei + 1] * 60 / time_resolution

            ll_intensity = ll_val * 60 / durations[d]
            ll_Ekin_i    = E_kin_i_vectorized(ll_hourly_intensities)
            ll_Ekin      = float(np.sum(ll_Ekin_i * ll_depth))
            ll_Re        = ll_Ekin * ll_intensity

            ll_vals.append(ll_val)
            ll_intensities.append(ll_intensity)
            ll_accums.append(ll_accum)
            ll_dates.append(ll_date)
            ll_starts.append(ll_start)
            ll_ends.append(ll_end)
            ll_Ekins.append(ll_Ekin)
            ll_Res.append(ll_Re)

        df_oe = pd.DataFrame({
            'year':               ll_yrs,
            'event_start':        ll_starts,
            'event_end':          ll_ends,
            'event_peak':         ll_dates,
            'prec_depth':         ll_vals,
            'intensity_per_hour': ll_intensities,
            'prec_accum':         ll_accums,
            'E_kin':              ll_Ekins,
            'erosivity_EU':       ll_Res,
            'erosivity_US':       [x * 10 for x in ll_Res],
        })
        dict_events[f"{durations[d]}"] = df_oe

    return dict_events

def E_kin_i(intensity):
    """
    Unit kinetic energy of rainfall [kJ m-2 mm-1] as a function of intensity [mm/h].

    Formula per DIN 19708:2017-08, based on Rogler & Schwertmann (1981).
    Equivalent to the USLE metric form from Foster et al. (1981) / Williams & Sheridan (1991) eq(3).
    Upper limit at 76.2 mm/h: raindrop size and terminal velocity approach a physical maximum
    at high intensities (Van Dijk et al., 2002).

    Parameters
    ----------
    intensity : float
        Rainfall intensity [mm/h].

    Returns
    -------
    float
        Unit kinetic energy [kJ m-2 mm-1].
    """
    if intensity < 0.05:
        return 0
    elif intensity >= 76.2:
        return 28.33 * 10**-3
    else:
        return (11.89 + 8.73 * np.log10(intensity)) * 10**-3


def E_kin_i_BrFr(intensity):
    """
    Unit kinetic energy [MJ ha-1 mm-1] after Brown & Foster (1987) -- US units.

    NOT used in the main pipeline. If substituted for E_kin_i it would produce
    erosivity in US units directly, breaking the EU-unit pathway and the x10 conversion.
    Kept here for reference / testing only.
    """
    return 0.29*(1-0.72*np.exp(-0.05*intensity))
    
def get_only_erosivity_events(df, accum_threshold=12.7, intensity_threshold=12.7, use_both_thresholds=True):
    """
    Filter erosivity events using the Wischmeier (1959, 1979) / Wischmeier & Smith (1978) criteria,
    also adopted by Rogler & Schwertmann (1981) and DIN 19708:2017-08.

    An event is erosive if either:
        (i)  total accumulated event depth  >= accum_threshold    [mm]  (default 12.7 mm = 0.5 in)
        (ii) peak window intensity          >= intensity_threshold [mm/h] (default 12.7 mm/h)

    The 12.7 mm/h default for criterion (ii) is IMax30 — the maximum 30-min intensity — which is
    the standard Wischmeier (1959) threshold.  Its value originates from the assumption that the
    marginal erosive event concentrates 6.35 mm (0.25 in) in 15 min with negligible rain outside
    that 15-min window.  Under this assumption, the 30-min rolling window always captures those
    6.35 mm, giving IMax30 = 6.35 * 60/30 = 12.7 mm/h.

    The same physical scenario observed at different accumulation window sizes would appear as:
        15-min window  →  IMax15 = 6.35 * 60/15 = 25.4 mm/h
        30-min window  →  IMax30 = 6.35 * 60/30 = 12.7 mm/h  (standard default)
        60-min window  →  IMax60 = 6.35 * 60/60 =  6.35 mm/h

    These are NOT interchangeable thresholds to swap by resolution — 12.7 mm/h for IMax30 is the
    fixed criterion.  The table only illustrates how the same assumed event translates across window
    sizes.  The 25.4 mm/h figure further assumes that all 6.35 mm falls uniformly over the full
    15 min, which is rarely the case in practice.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events_values. prec_depth and intensity_per_hour must correspond to the
        same accumulation window (e.g. both from durations=[30] for IMax30).
    accum_threshold : float, optional
        Minimum total event depth [mm]. Default 12.7 (Wischmeier 1959).
    intensity_threshold : float, optional
        Minimum peak window intensity [mm/h]. Default 12.7 (IMax30 equivalent of 6.35 mm/15 min).
    use_both_thresholds : bool, optional
        If True, apply criteria (i) OR (ii). If False, apply only criterion (ii). Default True.

    Returns
    -------
    filtered_df : pd.DataFrame
    """
    if use_both_thresholds:
        filtered_df = df[(df['intensity_per_hour'] >= intensity_threshold) | (df['prec_accum'] >= accum_threshold)]
    else:
        filtered_df = df[(df['intensity_per_hour'] >= intensity_threshold)]
        
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def find_optimal_thr_imax30(
    df_all_events, target_mean_annual, use_both_thresholds=False
):
    """
    Find the intensity threshold that minimises the difference between the
    mean annual erosive event count and target_mean_annual.

    The mean annual count is computed by grouping events by calendar year and
    averaging over all years present in df_all_events (years with zero events
    are included via reindex so the denominator is always the full record
    length). The objective is a step function of the threshold, so the global
    minimum is found by evaluating it at every unique intensity_per_hour value
    (O(n log n)).

    Typical use: match the mean annual count of 60-min erosivity events to
    the mean annual count of 5-min erosivity events by tuning thr_imax30.

    Parameters
    ----------
    df_all_events : pd.DataFrame
        Full event DataFrame before intensity filtering.
    target_mean_annual : float
        Desired mean annual number of erosivity events.
    use_both_thresholds : bool
        Passed through to get_only_erosivity_events.

    Returns
    -------
    thr_opt : float
        Optimal threshold [mm/h].
    achieved_mean_annual : float
        Mean annual event count at thr_opt.
    min_diff : float
        Residual |achieved_mean_annual - target_mean_annual|.
    """
    all_years = sorted(
        df_all_events['event_start'].dt.year.unique()
    )

    def _mean_annual(df_filtered):
        return (
            df_filtered
            .groupby(df_filtered['event_start'].dt.year)
            .size()
            .reindex(all_years, fill_value=0)
            .mean()
        )

    unique_thr = np.sort(df_all_events['intensity_per_hour'].unique())
    mean_annuals = np.array([
        _mean_annual(
            get_only_erosivity_events(
                df_all_events,
                intensity_threshold=float(t),
                use_both_thresholds=use_both_thresholds,
            )
        )
        for t in unique_thr
    ])
    diffs = np.abs(mean_annuals - target_mean_annual)
    # among ties pick the lowest threshold (keeps more events)
    idx_opt = int(np.where(diffs == diffs.min())[0][0])
    return (
        float(unique_thr[idx_opt]),
        float(mean_annuals[idx_opt]),
        float(diffs[idx_opt]),
    )


def mean_annual_erosivity(df, erosivity_col='erosivity_US'):
    """
    Compute mean annual erosivity (R-factor) from per-event erosivity.

    Annual erosivity is the sum of per-event EI30 within each calendar
    year. The R-factor is the mean of those annual sums over all years
    present in the dataset (Wischmeier & Smith 1978).

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_only_erosivity_events. Must contain an
        'event_start' datetime column and an erosivity column.
    erosivity_col : str, optional
        Name of the per-event erosivity column. Default 'erosivity_US'.

    Returns
    -------
    r_factor : float
        Mean annual erosivity [MJ mm ha-1 h-1 yr-1].
    annual : pd.Series
        Annual erosivity sums indexed by year.
    """
    all_years = sorted(df['event_start'].dt.year.unique())
    annual = (
        df.groupby(df['event_start'].dt.year)[erosivity_col]
        .sum()
        .reindex(all_years, fill_value=0)
    )
    r_factor = annual.mean()
    return r_factor, annual


def get_only_erosivity_events_Renard(df, accum_threshold=12.7, depth_threshold=6.35, time_resolution=None):
    """
    Filter erosivity events using the Renard et al. (1997) RUSLE criteria.

    An event is erosive if either:
        (i)  total accumulated event depth >= accum_threshold  [mm]  (default 12.7 mm)
        (ii) maximum 15-min accumulated depth >= depth_threshold [mm] (default 6.35 mm = 0.25 in)

    Criterion (ii) requires data at <= 15-min resolution so that prec_depth
    reflects a true 15-min window. If time_resolution > 15, this function raises
    a ValueError — use get_only_erosivity_events for coarser data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events_values with durations=[15].
    accum_threshold : float, optional
        Minimum total event depth [mm]. Default 12.7.
    depth_threshold : float, optional
        Minimum 15-min accumulated depth [mm]. Default 6.35 (Renard 1997).
    time_resolution : int or float, optional
        Data time step in minutes. Must be <= 15 to apply criterion (ii).

    Returns
    -------
    filtered_df : pd.DataFrame
    """
    if time_resolution is not None and time_resolution > 15:
        raise ValueError(
            f"time_resolution={time_resolution} min is too coarse for the Renard 6.35 mm/15-min "
            f"criterion. Use get_only_erosivity_events for data coarser than 15 min."
        )
    filtered_df = df[(df['prec_depth'] >= depth_threshold) | (df['prec_accum'] >= accum_threshold)]
    return filtered_df.reset_index(drop=True)

def boostrapping_erosivity_60min(df_erosivity, niter=1000, M=None):
    """
    Bootstrap annual erosivity statistics from a 60-min erosivity DataFrame.

    Resamples calendar years with replacement to estimate uncertainty in
    mean annual erosivity, event count, and mean intensity.

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of get_only_erosivity_events with erosivity_US_adj column added.
    niter : int, optional
        Number of bootstrap iterations. Default 1000.
    M : int, optional
        Number of years per bootstrap sample. Defaults to the number of unique
        years in the dataset.

    Returns
    -------
    df_bootstrap_summary : pd.DataFrame
        One row per bootstrap sample with columns: mean_annual_events,
        mean_annual_Imax, mean_rain_depth, average_annual_erosivity.
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

    
def boostrapping_erosivity_CPM_60min(df_erosivity, niter=1000, M=None, randy=None):
    """
    Bootstrap annual erosivity statistics, with optional external sample index (CPM use case).

    Same as boostrapping_erosivity_60min but accepts a pre-defined `randy` array so that
    bootstrap samples can be shared across multiple datasets (e.g. climate model ensembles).

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of get_only_erosivity_events with erosivity_US_adj column added.
    niter : int, optional
        Number of bootstrap iterations. Ignored if randy is provided. Default 1000.
    M : int, optional
        Years per bootstrap sample. Ignored if randy is provided. Defaults to
        the number of unique years in the dataset.
    randy : np.ndarray of int, optional
        Pre-defined sample index array, shape (niter, M). Values must be either:
        - 1-based consecutive integers (1..N) for fast index mapping, or
        - arbitrary integers mapped to the sorted unique years in df_erosivity.
        If provided, niter and M are ignored.

    Returns
    -------
    df_bootstrap_summary : pd.DataFrame
        One row per bootstrap sample with columns: mean_annual_events,
        mean_annual_Imax, mean_rain_depth, average_annual_erosivity.
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
