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


def get_events(
    data, dates, separation, min_rain,
    name_col='value', check_gaps=True,
    time_resolution=None, min_ev_dur=None,
):
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
        Column name for precipitation values when data is a DataFrame.
        Default 'value'.
    check_gaps : bool, optional
        If True, remove events whose start/end falls within `separation`
        hours of a data gap or the dataset boundary. Default True.
    time_resolution : int or float, optional
        Data time step [minutes]. Required when min_ev_dur is provided.
    min_ev_dur : int or float, optional
        Minimum event duration [minutes]. Events shorter than this are
        removed. If None, no duration filtering is applied. Intended for
        IDF analysis; not needed for erosivity (use get_only_erosivity_events
        for erosive event filtering instead).

    Returns
    -------
    arr_dates : np.ndarray, shape (N, 2)
        Array of (end, start) timestamp pairs for each retained event,
        ready to pass directly to get_events_values.
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
        split_points = np.where(time_diffs_above >= separation_ns)[0] + 1
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
                    
        for del_index in sorted(match_info, reverse=True):
            del consecutive_values[del_index]

    if min_ev_dur is not None:
        if time_resolution is None:
            raise ValueError(
                "time_resolution must be provided when min_ev_dur is set."
            )
        _, arr_dates, _ = remove_short(
            consecutive_values,
            time_resolution=time_resolution,
            min_ev_dur=min_ev_dur,
        )
        return arr_dates

    # No duration filter — convert list to arr_dates directly
    arr_dates = np.array([(ev[-1], ev[0]) for ev in consecutive_values])
    return arr_dates



def remove_short(list_events:list, time_resolution=None, min_ev_dur=None):
    """
    Remove precipitation events shorter than a minimum duration.

    Parameters
    ----------
    list_events : list of arrays
        Output of get_events — each element is an array of timestamps for
        one event.
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


def get_events_values(
    data, dates, arr_dates_oe, time_resolution=None, formula='rogler',
):
    """
    Compute per-event metrics for all accumulation windows supported by
    the data resolution.

    Valid windows are those whose length is a multiple of time_resolution,
    drawn from the standard set [5, 10, 15, 30, 60] minutes.  Windows
    shorter than the data step are never computed.  For example, 10-min
    data yields imax_10, imax_30, imax_60; 60-min data yields imax_60
    only.

    Erosivity (EI30) is NOT computed here.  Use get_erosivity() on the
    returned DataFrame, passing the imax column appropriate for your data.

    All kinetic energy formulas are normalized internally to
    [kJ m⁻² mm⁻¹] so that E_kin, erosivity_EU, and erosivity_US are
    consistent regardless of formula choice.

    Parameters
    ----------
    data : np.ndarray
        Full precipitation time series [mm per time step].
    dates : np.ndarray of np.datetime64
        Timestamps aligned with data.
    arr_dates_oe : np.ndarray, shape (N, 2)
        Event (end, start) pairs as returned by remove_short.
    time_resolution : int or float
        Data time step [minutes].  Required.
    formula : str, optional
        Kinetic energy formula to use.  Default 'rogler'.
        'rogler'     — Rogler & Schwertmann (1981) / DIN 19708:2017-08
                       log form, European calibration.
        'brown_foster' — Brown & Foster (1987), RUSLE standard,
                         exponential form.
        'mcgregor'   — McGregor et al. (1995), RUSLE2, exponential
                       form with steeper low-intensity decay.

    Returns
    -------
    df : pd.DataFrame
        One row per event.  Columns:
            year         : calendar year of event end
            event_start  : event start timestamp
            event_end    : event end timestamp
            event_depth    : total accumulated depth [mm]  — criterion (i)
            event_duration : event duration [h] (end − start + time_resolution)
            E_kin          : event kinetic energy [kJ m⁻²], integrated
                           per time step using E_kin_i_Rogler() or the
                           chosen formula; passed to
                           get_erosivity() to compute EI30
            imax_5       : peak 5-min intensity [mm/h]   — if resolution <= 5 min
            imax_10      : peak 10-min intensity [mm/h]  — if resolution <= 10 min
                           and 10 % resolution == 0
            imax_15      : peak 15-min intensity [mm/h]  — if resolution <= 15 min
                           and 15 % resolution == 0
            imax_30      : peak 30-min intensity [mm/h]  — criterion (ii) standard
            imax_60      : peak 60-min intensity [mm/h]
        Columns for windows not supported by the resolution are absent.
        EI30 (erosivity_US) is not included — call get_erosivity() next.
    """
    if time_resolution is None:
        raise ValueError("time_resolution must be provided.")

    _all_windows = [5, 10, 15, 30, 60]
    valid_windows = [
        w for w in _all_windows
        if w >= time_resolution and w % time_resolution == 0
    ]
    if not valid_windows:
        raise ValueError(
            f"time_resolution={time_resolution} min produces no valid "
            f"accumulation windows from {_all_windows}."
        )

    time_index = dates.reshape(-1)
    n_events = arr_dates_oe.shape[0]

    oe_end = arr_dates_oe[:, 0].astype("datetime64[ns]")
    oe_start = arr_dates_oe[:, 1].astype("datetime64[ns]")
    start_indices = np.searchsorted(time_index, oe_start)
    end_indices = np.searchsorted(time_index, oe_end)

    ll_yrs = [
        oe_end[i].astype("datetime64[Y]").item().year
        for i in range(n_events)
    ]

    _formulas = {
        'rogler':       E_kin_i_Rogler,
        'brown_foster': E_kin_i_BrFr,
        'mcgregor':     E_kin_i_McGregor,
    }
    if formula not in _formulas:
        raise ValueError(
            f"Unknown formula '{formula}'. "
            f"Choose from: {list(_formulas)}."
        )
    _e_kin_func = _formulas[formula]

    # Integer arithmetic avoids float accumulation errors in convolution
    data_int = np.round(data * 10000).astype(np.int64)

    E_kin_i_vec = np.vectorize(_e_kin_func)

    ll_starts = []
    ll_ends = []
    ll_accums = []
    ll_ekins = []
    # One list per valid window
    imax_lists = {w: [] for w in valid_windows}

    for i in range(n_events):
        si = start_indices[i]
        ei = end_indices[i]
        ll_starts.append(time_index[si])
        ll_ends.append(time_index[ei])

        if si == ei:
            depth_slice = np.array([data[si]])
            ll_accums.append(float(data[si]))
            for w in valid_windows:
                imax_lists[w].append(float(data[si]) * 60 / time_resolution)
        else:
            depth_slice = data[si:ei + 1]
            segment = data_int[si:ei + 1]
            ll_accums.append(float(np.sum(depth_slice)))
            for w in valid_windows:
                window_size = int(w / time_resolution)
                kernel = np.ones(window_size, dtype=np.int64)
                conv = np.convolve(segment, kernel, "same")
                peak_idx = np.nanargmax(conv)
                imax_lists[w].append(
                    conv[peak_idx] / 10000.0 * 60 / w
                )

        # E_kin: integrate unit kinetic energy over all time steps
        step_intensity = depth_slice * 60 / time_resolution
        e_kin = float(np.sum(E_kin_i_vec(step_intensity) * depth_slice))
        ll_ekins.append(e_kin)

    df = pd.DataFrame({
        'year': ll_yrs,
        'event_start': ll_starts,
        'event_end': ll_ends,
        'event_depth': ll_accums,
        'E_kin': ll_ekins,
    })
    df['event_duration'] = (
        (df['event_end'] - df['event_start'])
        / np.timedelta64(1, 'h')
        + time_resolution / 60
    )
    for w in valid_windows:
        df[f'imax_{w}'] = imax_lists[w]

    return df


def get_erosivity(df, imax_col='imax_30'):
    """
    Compute EI30 erosivity for each event.

    Multiplies the pre-computed kinetic energy (E_kin, stored in df by
    get_events_values) by the chosen peak intensity column.  The choice
    of imax column determines which accumulation window drives criterion
    (ii) and the EI30 product — pick the finest window your data
    resolution supports (typically imax_30 for <= 30-min data).

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events_values.  Must contain E_kin and imax_col.
    imax_col : str, optional
        Column to use as peak intensity for EI30.  Default 'imax_30'.

    Returns
    -------
    df : pd.DataFrame
        Copy of input with two new columns:
            erosivity_EU : E_kin × IMax  [kJ m⁻² mm h⁻¹]
            erosivity_US : same in US units [MJ mm ha⁻¹ h⁻¹]
                           (= erosivity_EU × 10)
    """
    if imax_col not in df.columns:
        raise ValueError(
            f"'{imax_col}' not found in DataFrame. "
            f"Available imax columns: "
            f"{[c for c in df.columns if c.startswith('imax_')]}"
        )
    df = df.copy()
    df['erosivity_EU'] = df['E_kin'] * df[imax_col]
    df['erosivity_US'] = df['erosivity_EU'] * 10
    return df

def E_kin_i_Rogler(intensity):
    """
    Unit kinetic energy after Rogler & Schwertmann (1981) [kJ m⁻² mm⁻¹].

    Log form calibrated to European rainfall, adopted verbatim by
    DIN 19708:2017-08. Default formula in get_events_values.

    Hard cap at 76.2 mm/h (= 28.33 × 10⁻³ kJ m⁻² mm⁻¹): above this
    intensity raindrop size and terminal velocity reach a physical
    maximum so kinetic energy no longer increases (van Dijk et al. 2002).
    The exponential formulas (Brown & Foster, McGregor) do not need an
    explicit cap as they asymptote naturally.

    Reference: Rogler, H. & Schwertmann, U. (1981). Erosivitaet der
    Niederschlaege und Isoerodentkarte Bayerns. J. Rural Eng. Developm.,
    22, 99-112.

    Parameters
    ----------
    intensity : float
        Rainfall intensity [mm/h].

    Returns
    -------
    float
        Unit kinetic energy [kJ m⁻² mm⁻¹].
    """
    if intensity < 0.05:
        return 0
    elif intensity >= 76.2:
        return 28.33e-3
    else:
        return (11.89 + 8.73 * np.log10(intensity)) * 1e-3


def E_kin_i_BrFr(intensity):
    """
    Unit kinetic energy after Brown & Foster (1987) [kJ m⁻² mm⁻¹].

    RUSLE standard (Renard et al. 1997). Exponential form fitted to
    measured drop-size distributions.

    Original publication units: MJ ha⁻¹ mm⁻¹.
    Normalized here to [kJ m⁻² mm⁻¹] (× 0.1) so E_kin integration
    and the erosivity_EU / erosivity_US pipeline are consistent across
    all formula choices.

    Reference: Brown, L.C. & Foster, G.R. (1987). Storm erosivity using
    idealized intensity distributions. Trans. ASAE, 30(2), 379-386.

    Parameters
    ----------
    intensity : float
        Rainfall intensity [mm/h].

    Returns
    -------
    float
        Unit kinetic energy [kJ m⁻² mm⁻¹].
    """
    # × 0.1 converts MJ ha⁻¹ mm⁻¹ → kJ m⁻² mm⁻¹
    # (1 MJ ha⁻¹ = 100 J m⁻² = 0.1 kJ m⁻²)
    # No hard cap needed — exponential form naturally plateaus at ~0.029
    return 0.29 * (1 - 0.72 * np.exp(-0.05 * intensity)) * 0.1


def E_kin_i_McGregor(intensity):
    """
    Unit kinetic energy after McGregor et al. (1995) [kJ m⁻² mm⁻¹].

    Variant of the Brown & Foster exponential form with a steeper
    decay coefficient (-0.082 vs -0.05), giving higher energy estimates
    at low intensities. Used in RUSLE2.

    Original publication units: MJ ha⁻¹ mm⁻¹.
    Normalized here to [kJ m⁻² mm⁻¹] (× 0.1) so E_kin integration
    and the erosivity_EU / erosivity_US pipeline are consistent across
    all formula choices.

    Reference: McGregor, K.C., Binger, R.L. & Bowie, A.J. (1995).
    Erosivity index values for northern Mississippi.
    Trans. ASAE, 38(4), 1039-1047.

    Parameters
    ----------
    intensity : float
        Rainfall intensity [mm/h].

    Returns
    -------
    float
        Unit kinetic energy [kJ m⁻² mm⁻¹].
    """
    # × 0.1 converts MJ ha⁻¹ mm⁻¹ → kJ m⁻² mm⁻¹
    # No hard cap needed — exponential form naturally plateaus at ~0.029
    return 0.29 * (1 - 0.72 * np.exp(-0.082 * intensity)) * 0.1
    
def get_only_erosivity_events(
    df, accum_threshold=12.7, intensity_threshold=12.7,
    imax_col='imax_30', use_both_thresholds=True,
):
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
        Output of get_events_values.
    accum_threshold : float, optional
        Minimum total event depth [mm] — criterion (i). Default 12.7.
    intensity_threshold : float, optional
        Minimum peak intensity [mm/h] — criterion (ii). Default 12.7
        (IMax30 threshold, Wischmeier 1959).
    imax_col : str, optional
        Column to use for criterion (ii). Default 'imax_30'. Use the
        finest window your data resolution supports.
    use_both_thresholds : bool, optional
        If True, apply criterion (i) OR (ii). If False, apply only
        criterion (ii). Default True.

    Returns
    -------
    filtered_df : pd.DataFrame
    """
    if imax_col not in df.columns:
        raise ValueError(
            f"'{imax_col}' not found. "
            f"Available: {[c for c in df.columns if c.startswith('imax_')]}"
        )
    if use_both_thresholds:
        filtered_df = df[
            (df[imax_col] >= intensity_threshold)
            | (df['event_depth'] >= accum_threshold)
        ]
    else:
        filtered_df = df[df[imax_col] >= intensity_threshold]

    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


def apply_rusle_split(
    df_erosivity, data, dates, time_resolution,
    imax_col='imax_30',
    accum_threshold=12.7, intensity_threshold=12.7,
    use_both_thresholds=True,
    formula='rogler',
):
    """
    Apply the Renard et al. (1997) RUSLE 6-hour / 1.27 mm splitting rule to
    an already-filtered set of erosivity events.

    RUSLE CRITERION
    ---------------
    Renard et al. (1997) define a storm separation rule stricter than the
    classic Wischmeier & Smith (1978) 6-hour dry-spell: a storm is split
    wherever any 6-hour window within it accumulates less than 1.27 mm
    (0.05 in).  The intent is to separate genuinely distinct sub-storms that
    happen to be connected by low-intensity drizzle, rather than requiring a
    complete dry gap.

    WHY APPLY ONLY TO EROSIVE EVENTS
    ---------------------------------
    If a parent event fails both erosivity criteria (depth < accum_threshold
    AND intensity < intensity_threshold), no sub-event produced by splitting
    can pass either — depth can only decrease after splitting, and the
    parent's imax is the maximum over all its sub-periods.  Therefore,
    applying Renard splitting before or after the erosivity filter gives
    identical results.  Filtering first limits the expensive split-and-
    recompute loop to ~1 000 erosive events instead of ~5 000+ raw events.

    WINDOW IMPLEMENTATION AND KNOWN LIMITATIONS
    --------------------------------------------
    The RUSLE criterion was defined for manual analysis and does not prescribe
    a specific sliding-window direction.  This implementation uses a
    bidirectional approach: for each wet time step the maximum of the trailing
    6-hour sum and the forward 6-hour sum is taken.  A step is flagged as a
    split point only if BOTH directions see < 1.27 mm, i.e. it is genuinely
    isolated from heavy rain in the past AND the future.

    Alternatives tested on 30 years of 5-min data at one Alpine station and
    compared against RIST 3.99 (which also implements Renard but in inches
    with undisclosed precision):

      trailing-only  — 890 erosive events (RIST: 962): over-splits because
                       early drizzle steps see no past rain and get falsely
                       flagged as split points.
      forward-only   — 888 erosive events: same problem at storm ends.
      bidirectional  — 959 erosive events: closest to RIST; residual
                       difference is unexplained and irresolvable without
                       RIST source code.

    The small remaining gap (959 vs 962) originates from RIST's internal
    inch-based arithmetic with unknown rounding, not from a bug here.
    The standard 6-hour dry-spell pipeline (get_events) gives 966 events,
    of which 4 extras vs RIST are fully explained by inch rounding at the
    12.8 mm / 12.8 mm/h threshold.

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of get_only_erosivity_events — the set of erosive events to
        split.  Must contain 'event_start' and 'event_end' columns.
    data : np.ndarray
        Full precipitation time series [mm per time step], aligned with
        dates.  Values below the drizzle threshold must already be zeroed.
    dates : np.ndarray of np.datetime64
        Timestamps aligned with data.
    time_resolution : float
        Data time step [minutes].
    imax_col : str, optional
        Intensity column for the erosivity criterion and EI30 product.
        Default 'imax_30'.
    accum_threshold : float, optional
        Minimum total event depth [mm] — criterion (i). Default 12.7.
    intensity_threshold : float, optional
        Minimum peak intensity [mm/h] — criterion (ii). Default 12.7.
    use_both_thresholds : bool, optional
        If True apply criterion (i) OR (ii); if False apply (ii) only.
        Default True.
    formula : str, optional
        Kinetic energy formula passed to get_events_values.
        Default 'rogler'.

    Returns
    -------
    df_split : pd.DataFrame
        Erosive events after Renard splitting and re-filtering, same format
        as the output of get_only_erosivity_events.
    """
    window_steps = int(6 / (time_resolution / 60))
    date_to_idx = {d: i for i, d in enumerate(dates)}

    s = pd.Series(data)
    trail = s.rolling(window=window_steps, min_periods=1).sum()
    fwd = s[::-1].rolling(window=window_steps, min_periods=1).sum()[::-1]
    full_roll = np.maximum(trail.to_numpy(), fwd.to_numpy())

    dates_ns = dates.astype('datetime64[ns]')
    wet_mask = data > 0

    sub_event_pairs = []
    for _, row in df_erosivity.iterrows():
        ev_start = np.datetime64(row['event_start'], 'ns')
        ev_end = np.datetime64(row['event_end'], 'ns')
        mask = wet_mask & (dates_ns >= ev_start) & (dates_ns <= ev_end)
        event_dates = dates[mask]

        if len(event_dates) == 0:
            continue

        # Bidirectional 6h rolling sum for this event's wet steps
        event_idx = np.array([date_to_idx[d] for d in event_dates])
        roll = full_roll[event_idx]
        wet = roll > 1.27

        # Split wherever the rolling sum drops below threshold
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

        for se in splits:
            if len(se) > 0:
                sub_event_pairs.append((se[-1], se[0]))

    if not sub_event_pairs:
        return df_erosivity.copy()

    arr_dates = np.array(sub_event_pairs)
    df_split = get_events_values(
        data=data, dates=dates,
        arr_dates_oe=arr_dates,
        time_resolution=time_resolution,
        formula=formula,
    )
    df_split = get_erosivity(df_split, imax_col=imax_col)
    df_split = get_only_erosivity_events(
        df_split,
        imax_col=imax_col,
        accum_threshold=accum_threshold,
        intensity_threshold=intensity_threshold,
        use_both_thresholds=use_both_thresholds,
    )
    return df_split


def get_mean_annual_stats(
    df,
    year_col='event_start',
    ei30_col='erosivity_US',
    depth_col=None,
    intensity_col=None,
    all_years=None,
):
    """
    Compute mean annual statistics from an erosivity event DataFrame.

    For each statistic the annual values are first summed (or counted)
    per calendar year, then averaged across years.  Years with zero
    events still contribute a zero to the mean (pass ``all_years`` to
    ensure a consistent denominator).

    Parameters
    ----------
    df : pd.DataFrame
        Event table.  Must contain ``year_col`` as a datetime column and
        ``ei30_col`` as numeric EI30 values.
    year_col : str
        Column holding event datetime used to extract the calendar year.
    ei30_col : str
        Column with per-event EI30 [MJ mm ha⁻¹ h⁻¹].
    depth_col : str or None
        Column with per-event accumulated depth [mm].  Optional.
    intensity_col : str or None
        Column with per-event peak intensity [mm/h].  Optional.
    all_years : array-like or None
        Complete list of years to include.  Years absent from ``df``
        are filled with zero.  If None, only years present in ``df``
        are used.

    Returns
    -------
    dict
        Keys: 'n_events', 'erosivity', and optionally 'depth',
        'intensity'.  Each value is itself a dict with:
            'mean'   – mean across years
            'std'    – standard deviation across years
            'annual' – pd.Series indexed by year
    """
    years = df[year_col].dt.year

    def _annual_sum(col):
        s = df.groupby(years)[col].sum()
        if all_years is not None:
            s = s.reindex(all_years, fill_value=0)
        return s

    def _annual_count():
        s = df.groupby(years)[ei30_col].count()
        if all_years is not None:
            s = s.reindex(all_years, fill_value=0)
        return s

    def _stat(series):
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'annual': series,
        }

    out = {
        'n_events': _stat(_annual_count()),
        'erosivity': _stat(_annual_sum(ei30_col)),
    }
    if depth_col is not None:
        # mean event depth per year, then mean across years
        s = df.groupby(years)[depth_col].mean()
        if all_years is not None:
            s = s.reindex(all_years)
        out['depth'] = _stat(s.dropna())
    if intensity_col is not None:
        # mean event IMax30 per year, then mean across years
        s = df.groupby(years)[intensity_col].mean()
        if all_years is not None:
            s = s.reindex(all_years)
        out['intensity'] = _stat(s.dropna())
    return out


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


def get_only_erosivity_events_Renard(df, accum_threshold=12.7, depth_threshold=6.35, time_resolution=None):
    """
    Filter erosivity events using the Renard et al. (1997) RUSLE criteria.

    An event is erosive if either:
        (i)  total accumulated event depth >= accum_threshold  [mm]  (default 12.7 mm)
        (ii) maximum 15-min accumulated depth >= depth_threshold [mm] (default 6.35 mm = 0.25 in)

    Criterion (ii) requires data at <= 15-min resolution so that window_depth
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
    filtered_df = df[(df['window_depth'] >= depth_threshold) | (df['event_depth'] >= accum_threshold)]
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
        mean_event_depth=('event_depth', 'mean'),
        sum_erosivity_US_adj=('erosivity_US_adj', 'sum')
    ).reset_index()
    
    # Then bootstrap sample years from yearly_agg, sum/mean accordingly
    bootstrap_summaries = []
    
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
    
        overall_means = sample_df[['N_events', 'mean_intensity_per_hour', 'mean_event_depth', 'sum_erosivity_US_adj']].mean()
    
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
        mean_event_depth=('event_depth', 'mean'),
        sum_erosivity_US_adj=('erosivity_US_adj', 'sum')
    ).reset_index()
    
    # Then bootstrap sample years from yearly_agg, sum/mean accordingly
    bootstrap_summaries = []
    
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
    
        overall_means = sample_df[['N_events', 'mean_intensity_per_hour', 'mean_event_depth', 'sum_erosivity_US_adj']].mean()
    
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
