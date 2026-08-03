# Regenerating the v3 figures (hpvsim_methods_manuscript)

The before/after review figures (fig1, fig5 dwelltime/partners, fig6 ASR) are rendered from the
committed frozen baselines — no sims needed. Both "before" (`results/v2.2.6_baseline/`) and "after"
(`results/v3.0.0_baseline/`) hold the plot-ready arrays (`fig1_data.npz`, `fig5_*_summary.csv`,
`fig6_asr.csv`).

- The review helper `render_methods.py` (in the `hpvsim_v23_migration_review` repo) renders all four
  from a baseline dir: `python render_methods.py --baseline results/v2.2.6_baseline --outdir <dir>`.
- The repo's own `plot_fig1.py` / `plot_fig56.py` produce the live-sim versions; `save_fig1_baseline*.py`
  / `save_fig56_baseline.py` freeze new baselines; `compare_baselines.py` overlays versions.

Run from this repo with the v3 venv (`.venv`, hpvsim 3.0.0). fig1 (natural-history params) is
bit-identical v2↔v3; fig6 ASR is ~4–5× higher in v3 (removal of a v2 multiscale cancer-deflation
bug — v3 is the correct, GLOBOCAN-validated level). See the review repo for the write-up.
