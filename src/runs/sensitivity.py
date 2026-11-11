"""Tornado (one parameter at a time, to its low and high) and structural switches.

Metric: world food saved, mean of the ten years, kcal per person per day
(grain and bran freed + animal food kept + carcasses), per scenario and bracket;
the first-year value is kept alongside. Writes
results/tornado.csv and results/switches.csv.
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import inputs as I
from src.model import params as P
from src.model.prepare import prepare
from src.runs.core import BRACKETS, COMPONENTS, population, run_pair
from src.runs.pooling import run_shared

SWITCHES = {
    "central": ({}, {}),
    "CP proxy": ({"protein_model": "cp_proxy"}, {}),
    "usual residue fed as a fixed mixture (extra residue still ranked)": ({"usual_residue": "mix"}, {}),
    "herd free to change size (grain kept)": ({}, {"herd": "free"}),
    "fixed herd, up to 2% output loss": ({}, {"output_floor": 0.02}),
    "fixed herd, no output floor": ({}, {"output_floor": None}),
    "allocated share reduced by half": ({"access_mult": 0.5}, {}),
    "allocated share reduced by three quarters": ({"access_mult": 0.25}, {}),
    "allocation: dairy first": ({}, {"allocation": "dairy first"}),
    "herd dynamics off (proportional rule)": ({}, {"herd_dynamics": False}),
    "herd strategy: keep the largest herd that survives": ({}, {"strategy": "survival"}),
    "herd strategy: most milk and meat eaten each year": ({}, {"strategy": "eaten"}),
    "no grain displacement (residues only fill deficits)": ({}, {"displace_grain": False}),
    "oilseed meals and other co-products removed too": ({}, {"coproducts_gone": True}),
    "radius 10 km": ({"access_radius_km": 10.0}, {}),
    "radius 200 km": ({"access_radius_km": 200.0}, {}),
    "no distance limit within a country": ({"no_reach_limit": True}, {}),
    "no distance limit, residue shared between countries": ({"share_between_countries": True}, {}),
    "more intense collection (10% left on the field everywhere)": ({"intense_collection": True}, {}),
    "other uses released for feed": ({"reserve_other_mult": 0.0}, {}),
    "10% kept for bedding in North America, Oceania, Europe and Russia": ({"bedding_reserve": True}, {}),
    "straw treated with urea or ammonia": ({"treatment": True}, {}),
    "treated straw, only where the year's mean is above 10 C": ({"treatment": True, "treatment_min_temp": 10.0}, {}),
    "knock-out: no intake ceiling": ({"no_intake_ceiling": True}, {}),
    "knock-out: no protein constraint": ({"no_protein_constraint": True}, {}),
    "knock-out: neither intake ceiling nor protein constraint": ({"no_intake_ceiling": True, "no_protein_constraint": True}, {}),
    "ARC/SCA efficiencies of ME use (Corbett and Freer 2003)": ({"efficiency_system": "arc"}, {}),
    "no cold stress": ({"no_cold_stress": True}, {}),
    "no production in freezing months": ({"no_production_when_freezing": True}, {}),
    "onset in the worst month": ({"onset": "worst"}, {}),
    "no residue standing at onset": ({"no_onset_stock": True}, {}),
    "onset stock never lost to sowing (the last twelve months' unused residue all standing)": ({"no_clearing": True}, {}),
    "all baled (no field decay or loss at sowing)": ({"baled_share": 1.0}, {}),
    "residue ratios held at pre-catastrophe values (no response to shocked yields)": ({"rpr_hold_baseline": True}, {}),
}
# accounting choices, derived from the central run: the share of the culled animals eaten (the central run eats all)
CARCASS_RECOVERY = {"culled animals: half eaten": 0.5, "culled animals: none eaten": 0.0}
LEVERS = ("treatment", "no_intake_ceiling", "no_protein_constraint", "treatment_min_temp")     # intervention-only levers: the no-intervention run keeps the central values


def metric(tg, theta, bname, b, switches):
    base_theta = {k: v for k, v in theta.items() if k not in LEVERS}
    if theta.get("share_between_countries"):                     # three intervention runs (runs/pooling.py); no levers
        if len(base_theta) < len(theta):
            raise ValueError("residue shared between countries does not combine with intervention-only levers")
        prep, ni, iv, fs = run_shared(tg, base_theta, b, **switches)
    else:
        prep = prepare(tg, base_theta)
        prep_iv = prepare(tg, theta) if any(k in theta for k in LEVERS) else None
        _, ni, iv, fs = run_pair(tg, bracket=b, prep=prep, prep_iv=prep_iv, **switches)
    pop = population().sum()
    comp = {k: fs[k].sum(axis=0) / pop / 365 for k in COMPONENTS}
    total = comp["grain"] + comp["milk"] + comp["meat"] + comp["carcass"]
    return dict(scenario=tg, bracket=bname, total=total.mean(), total_year1=total[0], grain=comp["grain"].mean(),
                animal=(comp["milk"] + comp["meat"]).mean(), carcass=comp["carcass"].mean(), carcass_year1=comp["carcass"][0],
                herd_iv_year3=iv["n"][:, :, 2].sum() / prep.n0.sum(), herd_ni_year3=ni["n"][:, :, 2].sum() / prep.n0.sum(),
                extra_residue_fed_ej=iv["fate"]["fed_extra"].sum(axis=0).mean() / 1e12)


def job(args):
    label, kind, theta, switches = args
    rows = []
    for tg in I.SCENARIOS:
        for bname, b in BRACKETS.items():
            rows.append(dict(label=label, kind=kind, **metric(tg, theta, bname, b, switches)))
    return rows


def tornado_jobs():
    jobs = [("central", "central", {}, {})]
    for k, (mode, lo, hi, *_r) in P.SCALARS.items():
        if k in P.NOT_SAMPLED:
            continue
        if k in P.POSITION_PARAMS:
            lo, hi = -1.0, 1.0
        jobs += [(k, "low", {k: lo}, {}), (k, "high", {k: hi}, {})]
    return jobs


def switch_jobs():
    return [(label, "switch", theta, sw) for label, (theta, sw) in SWITCHES.items()]


def write_tornado(out):
    pd.DataFrame(out).round(4).to_csv(I.RESULTS / "tornado.csv", index=False)


def write_switches(out):
    central = [r for r in out if r["label"] == "central"]
    for label, share in CARCASS_RECOVERY.items():
        for r in central:
            out.append(dict(r, label=label, carcass=share * r["carcass"], total=r["total"] - (1 - share) * r["carcass"],
                            total_year1=r["total_year1"] - (1 - share) * r["carcass_year1"]))
    sw = pd.DataFrame(out).round(4)
    sw.to_csv(I.RESULTS / "switches.csv", index=False)
    return sw


def report():
    sw = pd.read_csv(I.RESULTS / "switches.csv")
    pd.set_option("display.width", 250)
    print(sw[sw.scenario == 47].pivot(index="label", columns="bracket", values=["total", "grain", "animal", "extra_residue_fed_ej"]).round(1).to_string())
    t = pd.read_csv(I.RESULTS / "tornado.csv")
    t = t[(t.scenario == 47)].pivot_table(index="label", columns=["bracket", "kind"], values="total")
    print(t.round(1).to_string())


def main():
    workers = int(os.environ.get("RR_SENS_WORKERS", "4"))   # parallel processes; size them to RAM
    with Pool(workers) as pool:
        write_tornado([r for rows in pool.map(job, tornado_jobs(), chunksize=1) for r in rows])
    with Pool(workers) as pool:
        write_switches([r for rows in pool.map(job, switch_jobs()) for r in rows])
    report()


if __name__ == "__main__":
    main()
