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
    Values type    : Precipitation during interval
    Header rows    : 1
    Scan interval  : 5 minutes

RIST screenshots with saved settings are in fig/:
    RIST_setup_station.jpg         — station / gauge configuration
    RIST_setup_Rfactor_Imax30.jpg  — R-factor settings using IMax30
    RIST_setup_Rfactor_Imax15.jpg  — R-factor settings using IMax15

IMax30 vs IMax15 threshold derivation:
    The marginal erosive event concentrates 6.35 mm (0.25 in) in a
    short window. Observed at different accumulation window sizes:
        IMax15 = 6.35 * 60/15 = 25.4 mm/h
        IMax30 = 6.35 * 60/30 = 12.7 mm/h

Flagged records (flag > 0) are zeroed before export.

NOTE: running this script produces res/VE_0091_5min_RIST_input.csv
(~70 MB). That file is excluded from git to keep the repository small.
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
df = df.loc['1990':'2020']

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
