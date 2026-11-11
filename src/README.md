# src

The model and the code that produces the paper's results, imported as the package `src`
(`pip install -e .` from the repository root, or run from the root with `python -m src....`).

- `inputs.py`: paths and readers for the inputs in `data/`.
- `model/`: inputs per scenario (`prepare.py`), parameters and their ranges (`params.py`), ration search
  (`ration.py`), protein (`protein.py`), residue ratios (`ratios.py`), herds and food saved (`mechanism.py`).
- `runs/`: the analyses; each writes files in `results/` (see `results/README.md`).
- `figures/`: one module per data figure of the paper and supplement (Figs. 2 to 9, S1 to S7; the
  schematic Fig. 1 is a drawing and has no script here), written to `results/figures/`.
- `reach/`: the reach allocation of residue to animals on the 5 arcmin grid, for Fig. 2: `grid.py` builds the
  grid from the inputs downloaded by `scripts/download_inputs.py`, `access.py` puts residue and demand on it and
  `flow.py` solves the maximum flow. `access.py` also describes the reach tables in `data/`
  (`residue_access.csv`, `residue_access_crops.npz`) and the map arrays, which are distributed as inputs.
