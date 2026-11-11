# Results

The published results of the paper, as written by the runs in `src/runs/` (rerunning overwrites them;
`scripts/compare_results.py` compares a rerun with these). `manifest.json` records the checksums of the inputs,
code and results and the package versions used. Figures are written to `results/figures/`.

| File | Written by | Content |
|---|---|---|
| `core_world.csv`, `core_region.csv`, `core_country.csv` | `src.runs.core` | central run: food saved and its parts per year, world, region and country |
| `core_herds.csv`, `core_fate.csv`, `core_binding.csv`, `core_residue_crops.csv` | `src.runs.core` | herds, fate of the residue, binding limits, residue eaten by crop |
| `tornado.csv`, `switches.csv` | `src.runs.sensitivity` | one-at-a-time sensitivity and model choices (Fig. 9, Figs. S6 and S7) |
| `calibration_check.csv`, `calibration_summary.csv` | `src.runs.calibration` | pre-catastrophe calibration of protein requirements and intake caps |
| `urea_budget.csv` | `src.runs.urea_budget` | nitrogen of straw treatment |
| `residue_inventory.csv`, `residue_benchmark.csv` | `src.runs.residue_inventory` | residue per crop, and comparison with Smerald et al. (2023) |
| `mc_world*.csv`, `mc_country*.csv`, `mc_parameters*.csv`, `mc_draws*.npz` | `src.runs.sampling` | Monte Carlo, 500 draws; `_treated` with straw treatment |
