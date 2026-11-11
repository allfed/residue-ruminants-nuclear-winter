"""Parameter Monte Carlo: joint draws of the selected ranged inputs.

Scalars follow a PERT distribution on (low, central, high). Residue ratio,
ME, protein and NDF each take one position, PERT on (-1, 0, 1), that puts
every crop at the same point of its own range. The onset month is uniform
over the year. Writes results/mc_world.csv (percentiles per
scenario, bracket, year and component), results/mc_country.csv (ten-year
mean per country) and results/mc_draws.npz.

    python -m src.runs.sampling 500 default|treated|both

Each draw is saved in a checkpoint directory under work/mc_checkpoints/ as it
finishes; rerunning the command restores the completed draws, computes the
missing ones in RR_MC_WORKERS processes (default 4) and writes the summaries.
"""
import argparse
import os
import sys
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from src import inputs as I
from src.model import params as P
from src.model.mechanism import food_saved, simulate
from src.model.prepare import QUANT, prepare
from src.runs.core import BRACKETS, COMPONENTS, population

PCT = (2.5, 16, 50, 84, 97.5)


def pert(rng, low, mode, high, lam=4.0):
    if not low <= mode <= high:
        raise ValueError("PERT requires low <= mode <= high")
    if high <= low:
        return mode
    a = 1 + lam * (mode - low) / (high - low)
    b = 1 + lam * (high - mode) / (high - low)
    return low + rng.beta(a, b) * (high - low)


def draw(rng):
    theta = P.defaults()
    for k, (mode, lo, hi, *_r) in P.SCALARS.items():
        if k in P.NOT_SAMPLED or k in P.POSITION_PARAMS:
            continue
        theta[k] = pert(rng, lo, mode, hi)
    for pname in QUANT.values():
        theta[pname] = pert(rng, -1.0, 0.0, 1.0)
    theta["onset"] = int(rng.integers(1, 13))
    return theta


VARIANTS = {"default": {}, "treated": {"treatment": True}}
BOTH = "both"           # both arms of each draw, which share its no-intervention runs


def arms_of(variant):
    """The arms a command computes: one variant, or both."""
    if variant == BOTH:
        return tuple(VARIANTS)
    if variant not in VARIANTS:
        raise ValueError(f"Choose {', '.join(VARIANTS)} or {BOTH}")
    return (variant,)


def draw_theta(seed, variants):
    """The parameters of draw ``seed``: ``draw``, then, if an arm is treated, the treatment values from the rng's
    next draws. prepare reads the treatment values only for a treated theta, so untreated runs ignore them."""
    rng = np.random.default_rng(seed)
    theta = draw(rng)
    if any(VARIANTS[v] for v in variants):
        theta["treatment_values"] = {k: pert(rng, *v) for k, v in P.TREATMENT_RANGES.items()}
    return theta


def recorded(theta, variant):
    """The parameters recorded for one arm of a draw (checkpoints, mc_parameters*.csv): the scalars of theta and,
    for a treated arm, the nested treatment draws."""
    scal = {k: v for k, v in theta.items() if np.isscalar(v) and not isinstance(v, (str, bool))}
    if VARIANTS[variant]:
        scal.update({f"treatment_{k}": v for k, v in theta["treatment_values"].items()})
    return scal


def preps_for(tg, theta, variants):
    """Scenario tg prepared without the lever, and each arm's preparation for its intervention (None: the same)."""
    prep = prepare(tg, theta)
    return prep, {v: prepare(tg, {**theta, **VARIANTS[v]}) if VARIANTS[v] else None for v in variants}


def summary(prep, ni, iv, pop):
    """What a draw keeps of one scenario and bracket: food saved by component and year, world (kcal per person per
    day; 4,T), and by component and country (kcal per year, ten-year mean; 4,C); the herd with and without (T)."""
    fs = food_saved(prep, ni, iv)
    return (np.stack([fs[k].sum(axis=0) for k in COMPONENTS]) / pop.sum() / 365,
            np.stack([fs[k].mean(axis=1) for k in COMPONENTS]),
            iv["n"].sum(axis=(0, 1)) / prep.n0.sum(), ni["n"].sum(axis=(0, 1)) / prep.n0.sum())


def pair_arms(tg, bname, prep, preps_iv, pop):
    """core.run_pair for every arm of one scenario and bracket, with the no-intervention run solved once: it is the
    reference of each arm's intervention run, on the arm's preparation. Returns {variant: summary}."""
    b = BRACKETS[bname]
    ni = simulate(prep, intervention=False, concentrate_ratio=b)
    return {v: summary(prep, ni, simulate(prep_iv or prep, intervention=True, concentrate_ratio=b, reference=ni), pop)
            for v, prep_iv in preps_iv.items()}


def arms(seed, variants):
    """Draw ``seed`` for every arm in ``variants``: {variant: (recorded parameters, {(scenario, bracket): summary})}.
    The arms share the draw's parameters and no-intervention runs."""
    theta = draw_theta(seed, variants)
    pop = population()
    out = {v: {} for v in variants}
    for tg in I.SCENARIOS:
        prep, preps_iv = preps_for(tg, theta, variants)
        for bname in BRACKETS:
            for v, values in pair_arms(tg, bname, prep, preps_iv, pop).items():
                out[v][(tg, bname)] = values
    return {v: (recorded(theta, v), out[v]) for v in variants}


def one(args):
    """One arm of one draw: (recorded parameters, output)."""
    seed, variant = args
    return arms(seed, (variant,))[variant]


def unread(path):
    """Files under data/ that the model does not read: the map context, for figures, and the READMEs."""
    rel = path.relative_to(I.DATA)
    return rel.parts[0] == "map_context" or rel.suffix.lower() == ".md"


def hashed_files():
    """What the draws' numbers depend on: the model code, this module, core and inputs, the pinned environment, and
    every data file but the unread ones."""
    code = list((I.ROOT / "src/model").glob("*.py"))
    code += [I.ROOT / "src/runs/sampling.py", I.ROOT / "src/runs/core.py",
             I.ROOT / "src/inputs.py", I.ROOT / "environment.yml"]
    data = [p for p in I.DATA.rglob("*") if p.is_file() and not unread(p)]
    return sorted([p for p in code if p.exists()] + data)


def checkpoint_dir(variant):
    """Invalidate cached draws whenever their inputs, code or environment change."""
    digest = hashlib.sha256(f"{sys.version}|{np.__version__}|{pd.__version__}|{scipy.__version__}".encode())   # scipy ships HiGHS
    for path in hashed_files():
        digest.update(str(path.relative_to(I.ROOT)).encode())
        digest.update(path.read_bytes())
    directory = I.WORK / "mc_checkpoints" / f"{variant}-{digest.hexdigest()[:20]}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_checkpoint(path, seed, variant, result):
    scalars, output = result
    arrays = {"seed": seed, "variant": variant, "parameters": json.dumps(scalars)}
    for (tg, bracket), values in output.items():
        for component, value in zip(("world", "country", "herd_iv", "herd_ni"), values):
            arrays[f"{component}_{tg}_{bracket}"] = value
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_checkpoint(path, seed, variant):
    with np.load(path, allow_pickle=False) as data:
        if int(data["seed"]) != seed or str(data["variant"]) != variant:
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        output = {(tg, b): tuple(data[f"{k}_{tg}_{b}"] for k in ("world", "country", "herd_iv", "herd_ni"))
                  for tg in I.SCENARIOS for b in BRACKETS}
        return json.loads(str(data["parameters"])), output


def checkpointed_draw(args):
    """The arms of one draw still missing, each saved into its checkpoint directory."""
    seed, variants, directories = args
    results = arms(seed, variants)
    for variant, result in results.items():
        save_checkpoint(Path(directories[variant]) / f"{seed}.npz", seed, variant, result)
    return seed, results


def shown(path):
    path = Path(path)
    return path.relative_to(I.ROOT) if path.is_relative_to(I.ROOT) else path


def main(n_draws=500, variant="default"):
    variants = arms_of(variant)
    if n_draws < 1:
        raise ValueError("Choose a positive draw count")
    seeds = range(1000, 1000+n_draws)
    directories = {v: checkpoint_dir(v) for v in variants}
    cached = {v: {s: load_checkpoint(directories[v] / f"{s}.npz", s, v)
                  for s in seeds if (directories[v] / f"{s}.npz").exists()} for v in variants}
    for v in variants:
        print(f"{v}: {len(cached[v])}/{n_draws} draws restored; checkpoints in {shown(directories[v])}", flush=True)
    pending = [(s, tuple(v for v in variants if s not in cached[v]), {v: str(d) for v, d in directories.items()})
               for s in seeds if any(s not in cached[v] for v in variants)]
    if pending:
        with Pool(int(os.environ.get("RR_MC_WORKERS", "4"))) as pool:   # local workers; size them to RAM
            for done, (seed, results) in enumerate(pool.imap_unordered(checkpointed_draw, pending, chunksize=1), 1):
                for v, result in results.items():
                    cached[v][seed] = result
                if done % 25 == 0 or done == len(pending):
                    print(f"{variant}: {done}/{len(pending)} missing draws complete", flush=True)
    for v in variants:
        summarize(v, [cached[v][s] for s in seeds])


def summarize(variant, res):
    """Percentiles over the draws of one arm: results/mc_world*, mc_country*, mc_parameters* and mc_draws*."""
    tag = "" if variant == "default" else f"_{variant}"
    pop = population()
    iso = I.countries()
    rows, crow, save = [], [], {}
    for tg in I.SCENARIOS:
        for bname in BRACKETS:
            w = np.stack([r[1][(tg, bname)][0] for r in res])                 # (D,4,T)
            c = np.stack([r[1][(tg, bname)][1] for r in res])                 # (D,4,C)
            save[f"world_{tg}_{bname}"] = w
            save[f"herd_iv_{tg}_{bname}"] = np.stack([r[1][(tg, bname)][2] for r in res])
            save[f"herd_ni_{tg}_{bname}"] = np.stack([r[1][(tg, bname)][3] for r in res])
            series = {k: w[:, j, :] for j, k in enumerate(COMPONENTS)}
            series["animal"] = series["milk"] + series["meat"]
            series["total"] = series["grain"] + series["animal"] + series["carcass"]
            for k, v in series.items():
                for t in range(v.shape[1]):
                    rows.append(dict(scenario=tg, bracket=bname, component=k, year=t + 1, mean=v[:, t].mean(),
                                     **{f"p{q}": np.percentile(v[:, t], q) for q in PCT}))
                m = v.mean(axis=1)
                rows.append(dict(scenario=tg, bracket=bname, component=k, year=0, mean=m.mean(), **{f"p{q}": np.percentile(m, q) for q in PCT}))
            j = {k: i for i, k in enumerate(COMPONENTS)}
            tot = sum(c[:, j[k]] for k in ("grain", "milk", "meat", "carcass")) / np.where(pop > 0, pop, np.nan)[None, :] / 365
            for i, cc in enumerate(iso):
                crow.append(dict(scenario=tg, bracket=bname, iso3=cc, mean=tot[:, i].mean(), **{f"p{q}": np.percentile(tot[:, i], q) for q in PCT}))
    pd.DataFrame(rows).round(3).to_csv(I.RESULTS / f"mc_world{tag}.csv", index=False)
    pd.DataFrame(crow).round(3).to_csv(I.RESULTS / f"mc_country{tag}.csv", index=False)
    pd.DataFrame([r[0] for r in res]).round(5).to_csv(I.RESULTS / f"mc_parameters{tag}.csv", index=False)
    np.savez_compressed(I.RESULTS / f"mc_draws{tag}.npz", **save)
    print(f"{variant}:\n" + pd.DataFrame(rows).query("year == 0 and component == 'total'").round(1).to_string(index=False))


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m src.runs.sampling",
                                     description="Compute or restore the Monte Carlo draws and write the summaries.")
    parser.add_argument("n_draws", nargs="?", type=int, default=500)
    parser.add_argument("variant", nargs="?", default="default", choices=(*VARIANTS, BOTH))
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args.n_draws, args.variant)
