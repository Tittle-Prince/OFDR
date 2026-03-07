# Plotting Scripts Index

This file tracks all plotting scripts. Keep these scripts for future figure editing.

## Phase1 / Legacy

- `src/evaluate.py`
  - Regression figures for CNN vs PINN.
  - Outputs in `results/figures/`.

- `check/scripts/phase1_archive_and_plot.py`
  - Phase1 archive + publication-style figures.
  - Outputs in `check/results/phase1_report_*/figures/`.

## Phase2

- `src/phase2/summarize_phase2.py`
  - Baseline comparison chart.
  - Outputs in `check/result/phase2/summary/` (or configured `phase2.results_dir`).

## Phase3 / Phase3b

- `src/phase3/summarize_phase3.py`
  - Phase3 ablation chart (Baseline / Dilated / SE / Dilated+SE).
  - Outputs in `results/phase3/summary/`.

- `src/phase3/run_phase3b_unified_compare.py`
  - Unified Dataset_B figure set:
    - `comparison_plot.png`
    - `parity_plot.png`
    - `residual_plot.png`
  - Outputs in `results/phase3b/`.

- `src/phase3/plot_phase3b_error_zoom.py`
  - Error-zoom figure (`Pred-True` vs `True`).
  - Output: `results/phase3b/error_zoom_plot.png`.

## Notes

- Do not delete these scripts.
- If figure style changes, modify the script directly and rerun the command.
- Keep output directories separate by phase to avoid overwriting unrelated results.
