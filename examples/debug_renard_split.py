# -*- coding: utf-8 -*-
"""
Debug script — 2004-05-21 event (8.8 mm, in pyEr 6h but not RIST).

Passes via imax30 >= 12.7 mm/h, not depth. RIST does not report it.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyErosivity import get_events, get_events_values, get_erosivity

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, '..', 'res')

station_num = "VE_0091"
name_col = "vals"
min_rain = 0.1
imax_col = "imax_30"

data = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_5min_newflag.parguqet.gzip")
)
data['time'] = pd.to_datetime(data['time'])
data = data.set_index('time')
data.loc[data['flag'] > 0, name_col] = np.nan
data.loc[data[name_col] < min_rain, name_col] = 0
data[name_col] = data[name_col].fillna(0)

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

arr_dates = get_events(
    data=df_arr, dates=df_dates, separation=6, min_rain=min_rain,
    name_col=name_col, check_gaps=False, time_resolution=5.0,
)
df_events = get_events_values(
    data=df_arr, dates=df_dates,
    arr_dates_oe=arr_dates, time_resolution=5.0,
)
df_events = get_erosivity(df_events, imax_col=imax_col)

# Find the 2004-05-21 event
target_date = pd.Timestamp("2004-05-21").date()
mask = df_events['event_start'].dt.date == target_date
ev = df_events[mask]
print("2004-05-21 event:")
print(ev[['event_start', 'event_end', 'event_depth', 'event_duration',
          'imax_5', 'imax_15', 'imax_30', 'erosivity_US']].to_string(index=False))

# Plot with ±2h buffer
if len(ev):
    es = ev.iloc[0]['event_start']
    ee = ev.iloc[0]['event_end']
    plot_start = es - pd.Timedelta(hours=2)
    plot_end = ee + pd.Timedelta(hours=2)
    raw = data.loc[plot_start:plot_end, name_col]

    wet = raw[raw > 0]
    gaps = []
    for i in range(len(wet) - 1):
        dur = (wet.index[i+1] - wet.index[i]).total_seconds() / 3600
        if dur > 1.0:
            gaps.append((wet.index[i], wet.index[i+1], dur))

    print(f"\nDry gaps > 1h ({len(gaps)} found):")
    for gs, ge, dur in gaps:
        print(f"  {gs} → {ge}  ({dur:.2f} h)")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(raw.index, raw.values, width=pd.Timedelta(minutes=4),
           align='edge', color='steelblue', edgecolor='none')
    ax.axvline(es, color='tomato', linestyle='--',
               linewidth=1, label='event boundary (pyEr)')
    ax.axvline(ee, color='tomato', linestyle='--', linewidth=1)
    for gs, ge, dur in gaps:
        ax.axvspan(gs, ge, color='orange', alpha=0.3)
        ax.text(gs + (ge - gs) / 2, ax.get_ylim()[1] * 0.85,
                f"{dur:.1f}h", ha='center', va='top',
                fontsize=7, color='darkorange')
    ax.set_xlabel('Time')
    ax.set_ylabel('Depth [mm / 5 min]')
    ax.set_title(
        f"2004-05-21  |  depth={ev.iloc[0]['event_depth']:.1f} mm"
        f"  |  imax30={ev.iloc[0]['imax_30']:.1f} mm/h"
        f"  |  NOT in RIST"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
