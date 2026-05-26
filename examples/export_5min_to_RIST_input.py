# -*- coding: utf-8 -*-
"""
Export 5-min precipitation parquet to RIST-compatible input CSV.

Column layout (matches RIST field mapping):
    Month, Day, Year, Hour, Minute, Second, Gauge_mm

Configure RIST as:
    Delimiter      : Comma
    Field mapping  : Month=1, Day=2, Year=3, Hour=4, Minute=5,
                     Second=6, Gauge=7
    Input units    : Metric (mm)
    Values type    : Depth (not cumulative tips)
    Header rows    : 1

Flagged records (flag > 0) are zeroed before export.
"""

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, '..', 'res')

#%%
station_num = "VE_0091"
min_rain = 0.1

df = pd.read_parquet(
    os.path.join(_RES, f"{station_num}_5min_newflag.parguqet.gzip")
)
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df.loc['1990':'1991']

df.loc[df['flag'] > 0, 'vals'] = 0
df.loc[df['vals'] < min_rain, 'vals'] = 0
df['vals'] = df['vals'].fillna(0)

out = pd.DataFrame({
    'Month':  df.index.month,
    'Day':    df.index.day,
    'Year':   df.index.year,
    'Hour':   df.index.hour,
    'Minute': df.index.minute,
    'Second': 0,
    'Gauge_mm': np.round(df['vals'].values, 4),
})

out_path = os.path.join(
    _RES, f"{station_num}_5min_RIST_input.csv"
)
out.to_csv(out_path, index=False)
print(f"Exported {len(out):,} rows → {out_path}")
print(out.head(10).to_string(index=False))
