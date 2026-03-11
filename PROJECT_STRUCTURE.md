# Project Structure

This repository is now organized with clearer ownership for model, simulation, training, scripts, and results assets.

## Core Layout

```text
src/
  ofdr/
    models/        # Reusable model definitions
    simulation/    # Reusable simulation and dataset-building logic
    training/      # Reusable training utilities
  core/            # Legacy compatibility entrypoints (wrappers)
  phase2/          # Legacy compatibility entrypoints + phase scripts
  phase3/          # Legacy compatibility entrypoints + phase scripts
  phase4a/         # Legacy compatibility entrypoints + phase scripts

scripts/
  analysis/        # Analysis scripts
  figures/         # Figure generation scripts
  maintenance/     # Maintenance and archival scripts

results/
  phase1/
  phase2/
  phase3/
  phase3b/
  phase4a/
  phase4_array_tmp/
  phase4_checks/
  summaries/       # Aggregated plots/tables detached from run folders
```

## Compatibility Policy

- Existing import paths are preserved via lightweight wrappers:
  - `src/phase3/models.py -> src/ofdr/models/phase3_cnn.py`
  - `src/phase3/train_utils.py -> src/ofdr/training/phase3_train_utils.py`
  - `src/phase4a/array_simulator.py -> src/ofdr/simulation/phase4_array_simulator.py`
  - `src/phase4a/local_window.py -> src/ofdr/simulation/phase4_local_window.py`
  - Similar wrappers exist for Phase2 and PINN core files.
- Existing script entrypoints continue to run without path changes.

## Placement Rules (Going Forward)

- New reusable model code: `src/ofdr/models/`
- New reusable simulation code: `src/ofdr/simulation/`
- New reusable training code: `src/ofdr/training/`
- One-off experiment scripts: keep under corresponding `src/phase*/`
- Summary artifacts: place under `results/summaries/<topic>/`

