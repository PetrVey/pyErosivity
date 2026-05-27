# pyErosivity

A Python package for computing rainfall erosivity (R-factor / EI30) from precipitation time series at any temporal resolution.

## Overview

Rainfall erosivity quantifies the capacity of rain to detach and transport soil, and is the driving force in the USLE/RUSLE family of soil erosion models. pyErosivity implements the full pipeline from raw precipitation data to per-event and annual erosivity statistics:

1. **Event extraction** -- storms are separated by a minimum dry-spell duration (default 6 h, Wischmeier & Smith 1958, 1978) using `get_events`.

2. **Event filtering** -- events shorter than a minimum duration are discarded (`remove_short`).

3. **Per-event metrics** -- for each event and each accumulation window (e.g. 30 min for IMax30), a true sliding-window convolution computes peak accumulated depth and converts it to intensity. Kinetic energy and erosivity (EI30) are computed over the full event using the DIN 19708 / Rogler & Schwertmann (1981) formula.

4. **Erosivity event selection** -- `get_only_erosivity_events` implements the Wischmeier (1959, 1979) / Wischmeier & Smith (1978) / DIN 19708:2017-08 dual criterion: IMax30 >= 12.7 mm/h **or** total depth >= 12.7 mm.

4a. **Optional Renard within-storm splitting** -- `apply_rusle_split` takes the already-filtered erosive events and further splits any storm where a 6-hour window accumulates less than 1.27 mm (Renard et al. 1997 RUSLE rule), then re-applies the erosivity filter. Applying the split after filtering is mathematically equivalent to applying it before (a sub-event cannot exceed the parent's depth or intensity) but is faster.

5. **Temporal resolution correction** -- for coarser data (e.g. 60-min), IMax30 is underestimated. Williams & Sheridan (1991) first addressed this by equating I30 to the maximum hourly depth for 60-min records. `find_optimal_thr_imax30` generalises this idea by finding the intensity threshold that minimises the difference in event count relative to a high-resolution reference, following Fischer et al. (2018).

6. **Bootstrap uncertainty** -- `bootstrapping_erosivity_60min` and `bootstrapping_erosivity_CPM_60min` estimate uncertainty in annual erosivity by block-bootstrapping over calendar years.

## Output columns

`get_events_values` returns a single DataFrame. Each row is one event; the columns are:

| Column | Description | RUSLE criterion |
|---|---|---|
| `event_start` | Event start timestamp | — |
| `event_end` | Event end timestamp | — |
| `event_depth` | Total accumulated depth over the whole event [mm] | Criterion (i): `event_depth` >= 12.7 mm |
| `event_duration` | Event duration [h] | — |
| `E_kin` | Kinetic energy of the event [kJ m⁻²] | — |
| `imax_5` | Peak 5-min intensity [mm/h] — if resolution ≤ 5 min | — |
| `imax_10` | Peak 10-min intensity [mm/h] — if resolution ≤ 10 min | — |
| `imax_15` | Peak 15-min intensity [mm/h] — if resolution ≤ 15 min | — |
| `imax_30` | Peak 30-min intensity [mm/h] | Criterion (ii): `imax_30` >= 12.7 mm/h |
| `imax_60` | Peak 60-min intensity [mm/h] | — |
| `erosivity_EU` | E_kin × IMax [kJ m⁻² mm h⁻¹] — added by `compute_erosivity` | — |
| `erosivity_US` | Same in US units [MJ mm ha⁻¹ h⁻¹] (= erosivity_EU × 10) — added by `compute_erosivity` | — |

The RUSLE dual criterion applied by `get_only_erosivity_events` is:
**`event_depth` >= 12.7 mm OR `imax_30` >= 12.7 mm/h**

## Key references

### Erosivity methodology

- Wischmeier, W. H. & Smith, D. D. (1978). *Predicting Rainfall Erosion Losses.* USDA Agric. Handbook 537.
- Renard, K. G. et al. (1997). *Predicting Soil Erosion by Water (RUSLE).* USDA Agric. Handbook 703.
- Williams, R. G. & Sheridan, J. M. (1991). Effect of rainfall measurement time on EI calculation. *Trans. ASAE*, 34(2), 402-406.
- Fischer, F. K., Winterrath, T. & Auerswald, K. (2018). Temporal- and spatial-scale and positional effects on rain erosivity. *Hydrol. Earth Syst. Sci.*, 22, 6505-6518. https://doi.org/10.5194/hess-22-6505-2018
- DIN 19708:2017-08. Bodenbeschaffenheit — Ermittlung der Erosionsgefährdung von Böden durch Wasser mit Hilfe der ABAG. Beuth Verlag, Berlin.

### Kinetic energy formulas

- Rogler, H. & Schwertmann, U. (1981). Erosivitaet der Niederschlaege und Isoerodentkarte Bayerns. *J. Rural Eng. Developm.*, 22, 99-112. *(log form, European calibration; adopted by DIN 19708)*
- Brown, L. C. & Foster, G. R. (1987). Storm erosivity using idealized intensity distributions. *Trans. ASAE*, 30(2), 379-386. *(exponential form, RUSLE standard)*
- McGregor, K. C., Binger, R. L. & Bowie, A. J. (1995). Erosivity index values for northern Mississippi. *Trans. ASAE*, 38(4), 1039-1047. *(exponential form, RUSLE2)*
- van Dijk, A. I. J. M., Bruijnzeel, L. A. & Rosewell, C. J. (2002). Rainfall intensity–kinetic energy relationships: a critical literature appraisal. *J. Hydrology*, 261(1), 1-23. https://doi.org/10.1016/S0022-1694(02)00020-3 *(review; basis for 76.2 mm/h intensity cap)*

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pyErosivity_env
```

### 2. Install pyErosivity in editable mode

From the root of the repository:

```bash
pip install -e .
```

### 3. Run the examples

Example scripts are in the `examples/` folder and must be run from the repo root:

```bash
python examples/01_example_RISTvsPyErosivity_only_Imax30.py
python examples/02_example_depth_vs_imax.py
python examples/test_bootstrapping.py
python examples/test_bootstrapping_CPM.py
```