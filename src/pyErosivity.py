# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:25:32 2024

@author: Petr

pyErosivity: rainfall erosivity (R-factor / EI30) from precipitation time series.

Expected call order
-------------------
1. remove_incomplete_years(df, name_col)
   → df (cleaned), time_resolution [min]

2. get_events(data_arr, dates_arr, separation, min_rain)
   → arr_dates  shape (N, 2)  dtype datetime64  columns: [start, end]

3. get_events_values(data_arr, dates_arr, arr_dates, time_resolution)
   → df  columns: event_start, event_end, event_depth, E_kin,
                  event_duration, imax_5/10/15/30/60 (resolution-dependent)

4. compute_erosivity(df)
   → df + columns: erosivity_EU [kJ m⁻² mm h⁻¹], erosivity_US [MJ mm ha⁻¹ h⁻¹]

5. get_only_erosivity_events(df, imax_col, intensity_threshold, accum_threshold)
   → df  filtered to erosive events only

6. apply_rusle_split(df_erosivity, data_arr, dates_arr, time_resolution)  [optional]
   → df  after Renard 1.27 mm / 6 h sub-storm splitting and re-filtering

Zero values below the drizzle threshold (e.g. data[data < min_rain] = 0) BEFORE
calling get_events and pass the same zeroed array to apply_rusle_split.
"""
import pandas as pd
import numpy as np
from packaging.version import parse

def remove_incomplete_years(
    data_pr, name_col='value', nan_to_zero=True, tolerance=0.1
):
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
    # Step 1: get resolution from mode of all time diffs (robust to gaps at
    # the start or end of the record)
    time_res = pd.Series(
        data_pr.index[1:] - data_pr.index[:-1]
    ).mode()[0].total_seconds() / 60
    # Step 2: Resample by year and count non-NaN values per year
    if parse(pd.__version__) > parse("2.2"):
        yearly_valid = data_pr.resample('YE').apply(
            lambda x: x.notna().sum()
        )
    else:
        yearly_valid = data_pr.resample('Y').apply(
            lambda x: x.notna().sum()
        )
    # Step 3: Estimate expected length of yearly timeseries
    expected = pd.DataFrame(index=yearly_valid.index)
    expected["Total"] = 1440 / time_res * 365
    # Step 4: Calculate fraction of valid data per year
    valid_percentage = (yearly_valid[name_col] / expected['Total'])
    # Step 5: Identify years below the tolerance threshold
    years_to_remove = valid_percentage[
        valid_percentage < 1 - tolerance
    ].index
    # Step 6: Remove those years and optionally fill remaining NaNs
    data_cleaned = data_pr[
        ~data_pr.index.year.isin(years_to_remove.year)
    ]
    if nan_to_zero:
        data_cleaned.loc[:, name_col] = (
            data_cleaned[name_col].fillna(0)
        )
    time_resolution = time_res
    return data_cleaned, time_resolution


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
        Timestamps corresponding to data. When data is a pd.DataFrame,
        dates is derived from the index automatically and this argument
        is ignored.
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
        Dropped events produce no warning. Set check_gaps=False if you
        need to audit boundary losses yourself.
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
        Array of (start, end) timestamp pairs for each retained event,
        ready to pass directly to get_events_values.
    """
    if isinstance(data, pd.DataFrame):
        dates = np.array(data.index)
        data = np.array(data[name_col])
    elif not isinstance(data, np.ndarray):
        raise TypeError(
            f"data must be pd.DataFrame or np.ndarray, "
            f"got {type(data).__name__}."
        )

    dates = dates.astype("datetime64[ns]")

    if min_rain == 0:
        above_threshold_indices = np.where(data > min_rain)[0]
    else:
        above_threshold_indices = np.where(data >= min_rain)[0]

    if len(above_threshold_indices) == 0:
        return np.empty((0, 2), dtype='datetime64[ns]')

    # Vectorized event grouping: find gaps between consecutive wet steps,
    # then split at those gaps
    above_dates = dates[above_threshold_indices]
    time_diffs_above = np.diff(above_dates).astype(np.int64)
    separation_ns = int(separation * 3.6e12)
    split_points = np.where(time_diffs_above >= separation_ns)[0] + 1
    index_groups = np.split(above_threshold_indices, split_points)
    consecutive_values = [dates[group] for group in index_groups]

    if not consecutive_values:
        return np.empty((0, 2), dtype='datetime64[ns]')

    if check_gaps:
        # Remove event too close to dataset start
        if (
            (consecutive_values[0][0] - dates[0]).item()
            < (separation * 3.6e+12)
        ):
            consecutive_values.pop(0)

        # Remove event too close to dataset end
        if (
            (dates[-1] - consecutive_values[-1][-1]).item()
            < (separation * 3.6e+12)
        ):
            consecutive_values.pop()

        # Locate events that overlap data gaps
        time_diffs = np.diff(dates)
        # Time resolution from first step
        time_res = time_diffs[0]
        gap_indices_end = np.where(
            time_diffs > np.timedelta64(
                int(separation * 3.6e+12), 'ns'
            )
        )[0]
        gap_indices_start = (gap_indices_end + 1)

        match_info = []
        for gap_idx in gap_indices_end:
            end_date = dates[gap_idx]
            start_date = end_date - np.timedelta64(
                int(separation * 3.6e+12), 'ns'
            )
            temp_date_array = np.arange(start_date, end_date, time_res)
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(
                    np.isin(sub_array, temp_date_array)
                )[0]
                if match_indices.size > 0:
                    match_info.append(i)

        for gap_idx in gap_indices_start:
            start_date = dates[gap_idx]
            end_date = start_date + np.timedelta64(
                int(separation * 3.6e+12), 'ns'
            )
            temp_date_array = np.arange(start_date, end_date, time_res)
            for i, sub_array in enumerate(consecutive_values):
                match_indices = np.where(
                    np.isin(sub_array, temp_date_array)
                )[0]
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

    # No duration filter: convert list to arr_dates directly
    arr_dates = np.array([(ev[0], ev[-1]) for ev in consecutive_values])
    return arr_dates


def remove_short(list_events: list, time_resolution=None, min_ev_dur=None):
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
        Array of (start, end) timestamp pairs for each retained event.
    n_events_per_year : pd.DataFrame
        Count of retained events per calendar year.
    """
    if time_resolution is None or min_ev_dur is None:
        raise ValueError("time_resolution and min_ev_dur must both be provided.")

    if len(list_events) == 0:
        empty = np.empty((0, 2), dtype='datetime64[ns]')
        empty_df = pd.DataFrame(columns=['year', 'index'])
        return np.array([], dtype=bool), empty, empty_df

    # Normalise to np.datetime64 so there is one unified code path
    if isinstance(list_events[0][0], pd.Timestamp):
        list_events = [
            np.array([t.to_datetime64() for t in ev])
            for ev in list_events
        ]

    min_duration = np.timedelta64(int(min_ev_dur), "m")
    time_res = np.timedelta64(int(time_resolution), "m")

    ll_short = [
        (ev[-1] - ev[0]).astype("timedelta64[m]") + time_res >= min_duration
        for ev in list_events
    ]

    ll_dates = [
        (ev[0], ev[-1]) if keep else (np.nan, np.nan)
        for ev, keep in zip(list_events, ll_short)
    ]

    arr_vals = np.array(ll_short)[ll_short]
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

    Erosivity (EI30) is NOT computed here.  Use compute_erosivity() on the
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
        Event (start, end) pairs as returned by get_events or remove_short.
        The _oe suffix stands for Observed Events (OE), the term used in
        European erosivity literature for individual storm records.
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
                           compute_erosivity() to compute EI30
            imax_5       : peak 5-min intensity [mm/h]   — if resolution <= 5 min
            imax_10      : peak 10-min intensity [mm/h]  — if resolution <= 10 min
                           and 10 % resolution == 0
            imax_15      : peak 15-min intensity [mm/h]  — if resolution <= 15 min
                           and 15 % resolution == 0
            imax_30      : peak 30-min intensity [mm/h]  — used for criterion (ii) when imax_15 unavailable
            imax_60      : peak 60-min intensity [mm/h]
        Columns for windows not supported by the resolution are absent.
        event_duration is always present regardless of resolution.
        EI30 (erosivity_US) is not included. Call compute_erosivity() next.
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

    oe_start = arr_dates_oe[:, 0].astype("datetime64[ns]")
    oe_end = arr_dates_oe[:, 1].astype("datetime64[ns]")
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


def compute_erosivity(df):
    """
    Compute EI30 erosivity for each event.

    EI30 = E_kin × IMax30 always.  The intensity column is auto-detected
    from the DataFrame: imax_30 is used when present (all resolutions
    <= 30 min); imax_60 is used as fallback for hourly data where imax_30
    cannot be computed.  Both columns are produced by get_events_values.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events_values.  Must contain E_kin and at least one
        of imax_30 or imax_60.

    Returns
    -------
    df : pd.DataFrame
        Copy of input with two new columns:
            erosivity_EU : E_kin × IMax30  [kJ m⁻² mm h⁻¹]
            erosivity_US : same in US units [MJ mm ha⁻¹ h⁻¹]
                           (= erosivity_EU × 10)
    """
    if 'erosivity_US' in df.columns:
        raise ValueError(
            "erosivity_US already exists. compute_erosivity was called "
            "twice. Call it once on the output of get_events_values."
        )
    if 'imax_30' in df.columns:
        imax_col = 'imax_30'
    elif 'imax_60' in df.columns:
        imax_col = 'imax_60'
    else:
        available = [c for c in df.columns if c.startswith('imax_')]
        raise ValueError(
            f"Neither imax_30 nor imax_60 found. "
            f"Available imax columns: {available}"
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
    # No hard cap needed; exponential form naturally plateaus at ~0.029
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
    # No hard cap needed; exponential form naturally plateaus at ~0.029
    return 0.29 * (1 - 0.72 * np.exp(-0.082 * intensity)) * 0.1


def get_only_erosivity_events(
    df, accum_threshold=12.7, intensity_threshold=12.7,
    imax_col='imax_30', use_both_thresholds=True,
):
    """
    Filter erosivity events using the Wischmeier (1959, 1979) /
    Wischmeier & Smith (1978) criteria, also adopted by
    Rogler & Schwertmann (1981) and DIN 19708:2017-08.

    An event is erosive if either:
        (i)  total accumulated event depth >= accum_threshold [mm]
             (default 12.7 mm = 0.5 in)
        (ii) peak window intensity >= intensity_threshold [mm/h]
             (default 12.7 mm/h)

    The standard RUSLE criterion (ii) is IMax15 >= 25.4 mm/h
    (the maximum 15-min intensity). Its value originates from the
    assumption that the marginal erosive event concentrates
    6.35 mm (0.25 in) in exactly 15 min:
    IMax15 = 6.35 * 60/15 = 25.4 mm/h.

    The same physical scenario at different window sizes gives:
        15-min window  →  IMax15 = 6.35 * 60/15 = 25.4 mm/h
                          (standard RUSLE criterion)
        30-min window  →  IMax30 = 6.35 * 60/30 = 12.7 mm/h
                          (alternative, wider window)
        60-min window  →  IMax60 = 6.35 * 60/60 =  6.35 mm/h

    The 30-min window (default here, intensity_threshold=12.7) is an
    alternative that is looser in practice: it allows the same 6.35 mm
    to arrive over the full 30-min window, so it selects more events
    than IMax15 >= 25.4 mm/h on identical data.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events_values or compute_erosivity. Only
        event_depth and imax_col are required. compute_erosivity
        does not need to be called before this function.
    accum_threshold : float, optional
        Minimum total event depth [mm] — criterion (i). Default 12.7.
    intensity_threshold : float, optional
        Minimum peak intensity [mm/h] — criterion (ii). Default 12.7
        (IMax30 alternative threshold). For the standard IMax15 criterion
        use intensity_threshold=25.4 with imax_col='imax_15'.
    imax_col : str, optional
        Column to use for criterion (ii). Default 'imax_30'. Use the
        finest window your data resolution supports.
    use_both_thresholds : bool, optional
        If True, apply criterion (i) OR (ii). If False, apply only
        criterion (ii). Default True.

    Returns
    -------
    filtered_df : pd.DataFrame
        Subset of the input retaining only erosive events.  Same columns
        as the input (event_start, event_end, event_depth, E_kin,
        imax_*, and erosivity_* if compute_erosivity was called).
        Index is reset to 0-based integers.
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
    can pass either. Depth can only decrease after splitting, and the
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

      trailing-only:  890 erosive events (RIST: 962): over-splits because
                      early drizzle steps see no past rain and get falsely
                      flagged as split points.
      forward-only:   888 erosive events: same problem at storm ends.
      bidirectional:  959 erosive events: closest to RIST; residual
                      difference is unexplained and irresolvable without
                      RIST source code.

    The small remaining gap (959 vs 962) originates from RIST's internal
    inch-based arithmetic with unknown rounding, not from a bug here.
    The standard 6-hour dry-spell pipeline (get_events) gives 966 events,
    of which 4 extras vs RIST are fully explained by inch rounding at the
    12.8 mm / 12.8 mm/h threshold.

    Preconditions
    -------------
    ``data`` must already have sub-drizzle values zeroed (the same array
    you passed to get_events).  If raw data is passed instead, drizzle
    steps will be treated as wet and split points will be missed, silently
    producing too few sub-storms.

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of get_only_erosivity_events — the set of erosive events to
        split.  Must contain 'event_start' and 'event_end' columns.
    data : np.ndarray
        Full precipitation time series [mm per time step], aligned with
        dates.  Must be the same zeroed array passed to get_events.
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
                sub_event_pairs.append((se[0], se[-1]))

    if not sub_event_pairs:
        return df_erosivity.copy()

    arr_dates = np.array(sub_event_pairs)
    df_split = get_events_values(
        data=data, dates=dates,
        arr_dates_oe=arr_dates,
        time_resolution=time_resolution,
        formula=formula,
    )
    df_split = compute_erosivity(df_split)
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

        Aggregation note: n_events and erosivity are summed per year
        first, then averaged across years (sum-then-mean).  depth and
        intensity are averaged per year first (mean event value per
        year), then averaged across years (mean-of-means). They
        represent the typical event, not the annual total.
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
    df_all_events, target_mean_annual,
    imax_col='imax_30', use_both_thresholds=False,
):
    """
    Find the intensity threshold that minimises the difference between the
    mean annual erosive event count and target_mean_annual.

    The mean annual count is computed by grouping events by calendar year and
    averaging over all years present in df_all_events (years with zero events
    are included via reindex so the denominator is always the full record
    length). The objective is a step function of the threshold, so the global
    minimum is found by evaluating it at every unique imax_col value (O(n log n)).

    Typical use: match the mean annual count of 60-min erosivity events to
    the mean annual count of 5-min erosivity events by tuning thr_imax30.

    Parameters
    ----------
    df_all_events : pd.DataFrame
        Output of get_events_values, before intensity filtering.
    target_mean_annual : float
        Desired mean annual number of erosivity events.
    imax_col : str, optional
        Intensity column to sweep over. Default 'imax_30'.
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
    if imax_col not in df_all_events.columns:
        raise ValueError(
            f"'{imax_col}' not found. "
            f"Available: "
            f"{[c for c in df_all_events.columns if c.startswith('imax_')]}"
        )
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

    unique_thr = np.sort(df_all_events[imax_col].unique())
    mean_annuals = np.array([
        _mean_annual(
            get_only_erosivity_events(
                df_all_events,
                imax_col=imax_col,
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


def bootstrapping_erosivity_60min(
    df_erosivity, niter=1000, M=None,
    imax_col='imax_60', erosivity_col='erosivity_US',
):
    """
    Bootstrap annual erosivity statistics from an erosivity event DataFrame.

    Resamples calendar years with replacement to estimate uncertainty in
    mean annual erosivity, event count, intensity, and depth.

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of compute_erosivity + get_only_erosivity_events. Must contain
        event_start (datetime), event_depth, imax_col, and erosivity_col.
    niter : int, optional
        Number of bootstrap iterations. Default 1000.
    M : int, optional
        Number of years per bootstrap sample. Defaults to the number of
        unique years in the dataset.
    imax_col : str, optional
        Column with peak intensity [mm/h]. Default 'imax_30'.
    erosivity_col : str, optional
        Column with per-event erosivity [MJ mm ha⁻¹ h⁻¹].
        Default 'erosivity_US'. Pass a custom column name if you applied
        a temporal scale factor (e.g. 'erosivity_US_adj').

    Returns
    -------
    df_bootstrap_summary : pd.DataFrame
        One row per bootstrap sample with columns: mean_annual_events,
        mean_annual_Imax, mean_rain_depth, average_annual_erosivity.
    """
    df = df_erosivity.copy()
    years = df['event_start'].dt.year

    blocks = np.unique(years)
    if M is None:
        M = len(blocks)

    randy = np.random.choice(blocks, size=(niter, M), replace=True)

    yearly_agg = df.assign(_year=years).groupby('_year').agg(
        N_events=('event_depth', 'count'),
        mean_intensity_per_hour=(imax_col, 'mean'),
        mean_event_depth=('event_depth', 'mean'),
        sum_erosivity=(erosivity_col, 'sum'),
    ).reset_index().rename(columns={'_year': 'year'})

    bootstrap_summaries = []
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
        means = sample_df[[
            'N_events', 'mean_intensity_per_hour',
            'mean_event_depth', 'sum_erosivity',
        ]].mean()
        means.index = [
            'mean_annual_events', 'mean_annual_Imax',
            'mean_rain_depth', 'average_annual_erosivity',
        ]
        means['sample'] = f'sample_{i}'
        bootstrap_summaries.append(means)

    return pd.DataFrame(bootstrap_summaries).set_index('sample')


def compute_sf_annual_r(
    df_ref, df_target,
    ei30_col='erosivity_US',
    all_years=None,
):
    """
    Compute scaling factor (SF) from annual R-factor comparison.

    Pairs both DataFrames year by year. SF is the ratio of mean annual
    R-factors: SF = mean(R_ref) / mean(R_target). Multiply target event
    erosivity values by SF to correct the systematic bias.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference erosivity events. Must contain 'event_start' and
        ei30_col.
    df_target : pd.DataFrame
        Target erosivity events to correct.
    ei30_col : str, optional
        Erosivity column. Default 'erosivity_US'.
    all_years : list of int, optional
        Full year list. Years with zero erosive events are included
        (fill_value=0) so the denominator equals the record length.

    Returns
    -------
    sf : float
        Scaling factor (> 1 when target underestimates reference).
    r_ref : pd.Series
        Annual R-factor of reference indexed by year.
    r_target : pd.Series
        Annual R-factor of target indexed by year.
    """
    def _annual(df):
        years = df['event_start'].dt.year
        s = df.groupby(years)[ei30_col].sum()
        if all_years is not None:
            s = s.reindex(all_years, fill_value=0)
        return s

    r_ref = _annual(df_ref)
    r_target = _annual(df_target)
    if r_target.mean() == 0:
        raise ValueError(
            "Target mean annual R is zero. Cannot compute SF."
        )
    sf = float(r_ref.mean() / r_target.mean())
    return sf, r_ref, r_target


def compute_sf_per_event(
    df_ref, df_target,
    ei30_col='erosivity_US',
):
    """
    Compute scaling factor (SF) from per-event EI comparison.

    Events are matched by event_start date. Only events present in
    both datasets (inner join on date) are used. SF is the ratio of
    mean matched EI values: SF = mean(EI_ref) / mean(EI_target).

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference erosivity events. Must contain 'event_start' and
        ei30_col.
    df_target : pd.DataFrame
        Target erosivity events to correct.
    ei30_col : str, optional
        Erosivity column. Default 'erosivity_US'.

    Returns
    -------
    sf : float
        Scaling factor.
    ei_ref : pd.Series
        EI values of matched reference events.
    ei_target : pd.Series
        EI values of matched target events.
    n_matched : int
        Number of matched event pairs.
    """
    ref = df_ref[['event_start', ei30_col]].copy()
    tgt = df_target[['event_start', ei30_col]].copy()
    ref['_date'] = ref['event_start'].dt.date
    tgt['_date'] = tgt['event_start'].dt.date

    merged = pd.merge(
        ref[['_date', ei30_col]].rename(
            columns={ei30_col: '_ei_ref'}
        ),
        tgt[['_date', ei30_col]].rename(
            columns={ei30_col: '_ei_tgt'}
        ),
        on='_date', how='inner',
    )
    if merged['_ei_tgt'].mean() == 0:
        raise ValueError(
            "Target mean EI is zero. Cannot compute SF."
        )
    sf = float(merged['_ei_ref'].mean() / merged['_ei_tgt'].mean())
    return sf, merged['_ei_ref'], merged['_ei_tgt'], len(merged)


def bootstrapping_erosivity_CPM_60min(
    df_erosivity, niter=1000, M=None, randy=None,
    imax_col='imax_60', erosivity_col='erosivity_US',
):
    """
    Bootstrap annual erosivity statistics with optional pre-defined sample
    index (CPM / multi-dataset use case).

    Same as bootstrapping_erosivity_60min but accepts a pre-defined `randy`
    array so that bootstrap samples can be shared across multiple datasets
    (e.g. climate model ensembles where the same year-draw order is required
    for a fair comparison).

    Parameters
    ----------
    df_erosivity : pd.DataFrame
        Output of compute_erosivity + get_only_erosivity_events. Must contain
        event_start (datetime), event_depth, imax_col, and erosivity_col.
    niter : int, optional
        Number of bootstrap iterations. Ignored if randy is provided.
        Default 1000.
    M : int, optional
        Years per bootstrap sample. Ignored if randy is provided. Defaults
        to the number of unique years in the dataset.
    randy : np.ndarray of int, optional
        Pre-defined sample index array, shape (niter, M). Values must be
        either 1-based consecutive integers (1..N) for fast index mapping,
        or arbitrary integers mapped to the sorted unique years in
        df_erosivity. If provided, niter and M are ignored.
    imax_col : str, optional
        Column with peak intensity [mm/h]. Default 'imax_30'.
    erosivity_col : str, optional
        Column with per-event erosivity [MJ mm ha⁻¹ h⁻¹].
        Default 'erosivity_US'. Pass a custom column name if you applied
        a temporal scale factor (e.g. 'erosivity_US_adj').

    Returns
    -------
    df_bootstrap_summary : pd.DataFrame
        One row per bootstrap sample with columns: mean_annual_events,
        mean_annual_Imax, mean_rain_depth, average_annual_erosivity.
    """
    df = df_erosivity.copy()
    years = df['event_start'].dt.year
    blocks = np.sort(years.unique())

    if randy is None:
        if M is None:
            M = len(blocks)
        randy = np.random.choice(blocks, size=(niter, M), replace=True)
    else:
        if randy.dtype != np.int32:
            randy = randy.astype(np.int32)

        unique_randy_vals = np.unique(randy)
        n_randy_vals = len(unique_randy_vals)

        if len(blocks) != n_randy_vals:
            raise ValueError(
                f"Mismatch: randy has {n_randy_vals} unique values "
                f"but df_erosivity has {len(blocks)} unique years."
            )

        # Fast path for clean 1-based consecutive indices (1..N)
        is_1_based = np.array_equal(
            unique_randy_vals, np.arange(1, n_randy_vals + 1)
        )
        if is_1_based:
            randy = blocks[randy - 1]
        else:
            mapping = dict(zip(unique_randy_vals, blocks))
            randy = np.vectorize(mapping.get)(randy)

    yearly_agg = df.assign(_year=years).groupby('_year').agg(
        N_events=('event_depth', 'count'),
        mean_intensity_per_hour=(imax_col, 'mean'),
        mean_event_depth=('event_depth', 'mean'),
        sum_erosivity=(erosivity_col, 'sum'),
    ).reset_index().rename(columns={'_year': 'year'})

    bootstrap_summaries = []
    for i, sampled_years in enumerate(randy, 1):
        sample_df = yearly_agg[yearly_agg['year'].isin(sampled_years)]
        means = sample_df[[
            'N_events', 'mean_intensity_per_hour',
            'mean_event_depth', 'sum_erosivity',
        ]].mean()
        means.index = [
            'mean_annual_events', 'mean_annual_Imax',
            'mean_rain_depth', 'average_annual_erosivity',
        ]
        means['sample'] = f'sample_{i}'
        bootstrap_summaries.append(means)

    return pd.DataFrame(bootstrap_summaries).set_index('sample')
