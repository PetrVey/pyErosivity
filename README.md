# pyErosivity

A Python package for computing rainfall erosivity (R-factor / EI30) from precipitation time series at any temporal resolution.

## Overview

Rainfall erosivity quantifies the capacity of rain to detach and transport soil, and is the driving force in the USLE/RUSLE family of soil erosion models. pyErosivity implements the full pipeline from raw precipitation data to per-event and annual erosivity statistics:

1. **Event extraction** -- storms are separated by a minimum dry-spell duration (default 6 h, Wischmeier & Smith 1958, 1978). Two extraction methods are available:
   - `get_events` -- standard separation by dry gap
   - `get_events_Renard_RUSLE` -- RUSLE splitting rule: a gap of < 1.27 mm in any 6-hour window splits a storm into two events (Renard et al. 1997)

2. **Event filtering** -- events shorter than a minimum duration are discarded (`remove_short`).

3. **Per-event metrics** -- for each event and each accumulation window (e.g. 30 min for IMax30), a true sliding-window convolution computes peak accumulated depth and converts it to intensity. Kinetic energy and erosivity (EI30) are computed over the full event using the DIN 19708 / Rogler & Schwertmann (1981) formula.

4. **Erosivity event selection** -- two filter functions implement the main literature definitions:
   - `get_only_erosivity_events` -- Wischmeier (1959, 1979) / Wischmeier & Smith (1978), also DIN 19708:2017-08: IMax30 >= 12.7 mm/h **or** total depth >= 12.7 mm
   - `get_only_erosivity_events_Renard` -- Renard et al. (1997) RUSLE: total depth >= 12.7 mm **or** max 15-min depth >= 6.35 mm (requires <= 15-min data)

5. **Temporal resolution correction** -- for coarser data (e.g. 60-min), IMax30 is underestimated. `find_optimal_thr_imax30` finds the intensity threshold that minimises the difference in event count relative to a high-resolution reference, following the reasoning in Fischer et al. (2018).

6. **Bootstrap uncertainty** -- `boostrapping_erosivity_60min` and `boostrapping_erosivity_CPM_60min` estimate uncertainty in annual erosivity by block-bootstrapping over calendar years.

## Key references

- Wischmeier, W. H. & Smith, D. D. (1978). *Predicting Rainfall Erosion Losses.* USDA Agric. Handbook 537.
- Rogler, H. & Schwertmann, U. (1981). Erosivitaet der Niederschlaege und Isoerodentkarte Bayerns. *J. Rural Eng. Developm.*, 22, 99-112.
- Renard, K. G. et al. (1997). *Predicting Soil Erosion by Water (RUSLE).* USDA Agric. Handbook 703.
- Williams, R. G. & Sheridan, J. M. (1991). Effect of rainfall measurement time on EI calculation. *Trans. ASAE*, 34(2), 402-406.
- Fischer, F. K., Winterrath, T. & Auerswald, K. (2018). Temporal- and spatial-scale and positional effects on rain erosivity. *Hydrol. Earth Syst. Sci.*, 22, 6505-6518. https://doi.org/10.5194/hess-22-6505-2018

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
python examples/test_erosivity.py
python examples/test_bootstrapping.py
python examples/test_bootstrapping_CPM.py
```