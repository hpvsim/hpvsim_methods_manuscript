# Changelog

## 2026-04-18 — HPVsim v2.2.6 lift

- Added save scripts (`save_fig1_baseline.py`, `save_fig56_baseline.py`) that freeze plot-ready arrays to versioned baseline dirs (`results/v*_baseline/`).
- Extracted the original v1.0.0 paper values from committed `.obj` artifacts into `results/v1.0.0_published/` for cross-version comparison.
- Added `compare_baselines.py` for side-by-side comparison figures across baseline dirs.
- Updated `.gitignore` to exclude large simulation outputs from versioned subdirs.

## 2023 — Original publication

- Code for reproducing Figures 1, 4, 5, and 6 of the HPVsim methods manuscript, paired with `hpvsim==1.0.0`.
