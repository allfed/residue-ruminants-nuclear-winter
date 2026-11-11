# Feeding ruminants unused crop residues to free grain for people in abrupt sunlight reduction scenarios

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Testing](https://github.com/allfed/residue-ruminants-nuclear-winter/actions/workflows/testing.yml/badge.svg)

Code and data for the paper of the same title (Blouin et al.): how much
human-edible food feeding unused crop residue to ruminants could save after an abrupt sunlight reduction.

The model covers a nuclear winter with 16, 47 or 150 Tg of soot. For 169 countries over ten years, it follows
residue production and how much of it animals can reach, ruminant energy, protein and fiber requirements, cold
stress and herd sizes, and reports the grain freed and the milk and meat kept. A parameter Monte Carlo and a
one-at-a-time sensitivity analysis give the ranges.

## Installation

```bash
conda env create -f environment.yml
conda activate residue-ruminants
pip install -e .
```

The code was run on Linux with Python 3.12 and the pinned packages in `environment.yml` (the same pins are in
`requirements.txt`). Run every command from the repository root.

## Repository structure

```
data/      every input the runs read (data/README.md describes each file)
src/       the model and the code that produces the results (src/README.md)
  model/     the model itself
  runs/      the analyses, each writing files in results/
  figures/   one module per data figure (Figs. 2 to 9, S1 to S7), written to results/figures/
  reach/     the reach allocation on the 5 arcmin grid, for Fig. 2 (the reach tables in data/ are inputs)
scripts/   reproduce.sh, compare_results.py, download_inputs.py
results/   the published results and their checksums (results/README.md)
tests/     fast unit tests
work/      created when you run things, not tracked: a copy of the published results to compare with, logs,
           Monte Carlo checkpoints, and the downloads and grid for Fig. 2
```

## Reproduce the results

```bash
WORKERS=8 bash scripts/reproduce.sh
```

The script keeps a copy of the published results in `work/published_results/`, reruns every analysis, redraws
every data figure, then compares each results file with the published one (identical, or within rounding). It is
restartable: finished stages are skipped, and Monte Carlo draws are saved as they finish. Each stage can also be
run alone, for example `python -m src.runs.core` for the central run; `results/README.md` lists which module
writes which file.

**Time and memory.** On a 16-core desktop with 24 parallel processes, the central run took 18 minutes, the
sensitivity analysis 1.4 hours and the Monte Carlo (500 draws, with and without straw treatment) 33 hours,
about 800 process-hours in all. Choose `WORKERS` by memory rather than
cores: allow about 1 GB per process. For a quick test, `N_DRAWS=20 bash scripts/reproduce.sh` runs everything
with 20 draws; the Monte Carlo files then differ from the published ones.

## Figure 2 for any region

Figure 2 shows the reach allocation cell by cell on the native 5 arcmin grid (Spain, year 4 after 47 Tg, grain
kept). The grid is not distributed: download its public inputs (FAO GLW4 2020, MapSPAM 2020 V1r0, Natural Earth
5.1.1; about 177 MB, each file checked against the one used for the paper), build it, then draw the figure.
Harvard Dataverse asks every downloader of MapSPAM for an email address and an institution: give them with
`--email` and `--institution`, or download the two archives in a browser (see the script's docstring).

```bash
python scripts/download_inputs.py grid
python -m src.reach.grid
python -m src.figures.fig2_reach_inputs
python -m src.figures.fig2_reach_inputs --window 72 88 18 31 --circle-at 79 26.5 \
    --scenario 150 --year 2 --bracket kept --name fig2_india_150Tg_y2      # India, 150 Tg, year 2
```

## Paper

Blouin et al., *Feeding ruminants unused crop residues to free grain for people in abrupt sunlight reduction
scenarios*. Please cite the paper if you use this code or data.

## License

The code and the data derived by the authors are under the Apache License 2.0 (`LICENSE`). Data and values from
third-party sources (FAOSTAT, MapSPAM, GLW4, Feedipedia, Smerald et al., Xia et al. and others) are not
relicensed and remain under their sources' terms: `NOTICE` lists each source, its licence and the files that
draw on it.
