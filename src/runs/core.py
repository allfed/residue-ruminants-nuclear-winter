"""Central runs: 16, 47, 150 Tg, both cases for the feed grain (grain kept, grain gone).

Writes to results/:
  core_world.csv      per scenario, bracket, year: the components of food saved
                      (kcal per person per day, world), herd, milk and meat paths
  core_country.csv    the same per country (kcal per person per day of the country)
  core_region.csv     per GLEAM region
  core_fate.csv       residue fate, world, EJ ME per year
  core_residue_crops.csv  residue per crop, world, PJ ME per year: produced, fed with
                      and without the intervention, extra residue fed
  core_herds.csv      herd heads and production level per herd, world
  core_binding.csv    binding constraint per country, herd (year 2) and the country's
                      dominant one
"""
import numpy as np
import pandas as pd

from src import inputs as I
from src.model.mechanism import BINDING, RES_CROPS, food_saved, simulate
from src.model.prepare import no_shock, prepare

BRACKETS = {"kept": 1.0, "gone": 0.0}      # share of the pre-catastrophe grain and bran ration still fed; co-products stay in both
COMPONENTS = ("grain", "milk", "meat", "carcass")                   # grain is the grain and the bran


def run_pair(tg, theta=None, bracket=1.0, prep=None, prep_iv=None, **switches):
    """No-intervention and intervention runs; returns (prep, ni, iv, saved)."""
    prep = prep or prepare(tg, theta)
    sim_kw = {k: v for k, v in switches.items() if k in ("herd_dynamics", "strategy", "coproducts_gone", "tol", "scan")}
    ni = simulate(prep, intervention=False, concentrate_ratio=bracket, **sim_kw)
    iv = simulate(prep_iv or prep, intervention=True, concentrate_ratio=bracket, reference=ni, **switches)
    return prep, ni, iv, food_saved(prep, ni, iv)


def population():
    pop = I.population()
    return pop.fillna(0.0).to_numpy()


def main():
    pop = population()
    iso = I.countries()
    reg = np.array([I.regions()[c] for c in iso])
    world, country, region, fate, herds, binding, crops = [], [], [], [], [], [], []
    for tg in (0,) + tuple(I.SCENARIOS):
        for bname, b in BRACKETS.items():
            prep, ni, iv, fs = run_pair(tg, bracket=b, prep=no_shock(prepare(47)) if tg == 0 else None)
            for t in range(I.N_YEARS):
                row = dict(scenario=tg, bracket=bname, year=t + 1)
                for k in COMPONENTS:
                    row[k] = fs[k][:, t].sum() / pop.sum() / 365
                for name, r in (("ni", ni), ("iv", iv)):
                    row[f"herd_{name}"] = r["n"][:, :, t].sum() / prep.n0.sum()
                    row[f"milk_{name}"] = r["milk"][:, :, t].sum() / pop.sum() / 365
                    row[f"meat_{name}"] = r["meat"][:, :, t].sum() / pop.sum() / 365
                    row[f"carcass_{name}"] = r["carcass"][:, :, t].sum() / pop.sum() / 365
                    row[f"conc_ej_{name}"] = r["conc_used"][:, :, t].sum() / 1e12
                    row[f"coproduct_ej_{name}"] = r["cop_used"][:, :, t].sum() / 1e12
                    row[f"residue_ej_{name}"] = r["res_used"][:, :, t].sum() / 1e12
                    row[f"grass_ej_{name}"] = r["grass_used"][:, :, t].sum() / 1e12
                row["reach_rounds_iv"] = int(iv["access_rounds"][t])                        # herd and reach iterations (2.5)
                row["reach_unsettled_iv"] = int((~iv["access_converged"][:, t]).sum())        # countries offered the smallest reach seen
                world.append(row)
                f = iv["fate"]
                produced = prep.extras["e_res"][:, 0 if tg == 0 else t + 1, :].sum()
                fate.append(dict(scenario=tg, bracket=bname, year=t + 1, produced=produced / 1e12,
                                 feedable_flow=f["pool"][:, t].sum() / 1e12, stock_in=f["stock_in"][:, t].sum() / 1e12,
                                 fed=f["fed"][:, t].sum() / 1e12, fed_extra=f["fed_extra"][:, t].sum() / 1e12,
                                 fed_without=ni["res_used"][:, :, t].sum() / 1e12,
                                 extra_flow=f["extra_flow"][:, t].sum() / 1e12,
                                 usual_released=f["usual_released"][:, t].sum() / 1e12,
                                 unused_usual=f["unused_usual"][:, t].sum() / 1e12,
                                 carried=f["carried"][:, t].sum() / 1e12, decayed=f["decayed"][:, t].sum() / 1e12,
                                 stock_quality_change=f["stock_quality_change"][:,t].sum()/1e12,
                                 **{key:f[key][:,t].sum()/1e9 for key in ("extra_flow_dm","usual_released_dm","stock_in_dm","fed_extra_dm","carried_dm","decayed_dm")}))
                produced_k = prep.extras["e_res"][:, 0 if tg == 0 else t + 1, :].sum(axis=0) / 1e9
                fed_k, extra_k = f["fed_by_crop"][:, t].sum(axis=0) / 1e9, f["fed_extra_by_crop"][:, t].sum(axis=0) / 1e9
                without_k = ni["res_crop_used"][:, :, t].sum(axis=(0, 1)) / 1e9
                for k, code in enumerate(RES_CROPS):
                    crops.append(dict(scenario=tg, bracket=bname, year=t + 1, crop=code, produced_pj=produced_k[k],
                                      fed_pj=fed_k[k], fed_without_pj=without_k[k], extra_fed_pj=extra_k[k]))
                for h, hd in enumerate(I.HERDS):
                    herds.append(dict(scenario=tg, bracket=bname, year=t + 1, herd=hd,
                                      heads_ni=ni["n"][:, h, t].sum(), heads_iv=iv["n"][:, h, t].sum(), heads_prewar=prep.n0[:, h].sum(),
                                      cull_ni=ni["cull"][:, h, t].sum(), cull_iv=iv["cull"][:, h, t].sum(),
                                      p_ni=(ni["n"][:, h, t] * ni["p"][:, h, t]).sum() / max(ni["n"][:, h, t].sum(), 1),
                                      p_iv=(iv["n"][:, h, t] * iv["p"][:, h, t]).sum() / max(iv["n"][:, h, t].sum(), 1)))
                for r in sorted(set(reg)):
                    m = reg == r
                    rr = dict(scenario=tg, bracket=bname, year=t + 1, region=r)
                    for k in COMPONENTS:
                        rr[k] = fs[k][m, t].sum() / max(pop[m].sum(), 1) / 365
                    rr["herd_ni"] = ni["n"][m, :, t].sum() / max(prep.n0[m].sum(), 1)
                    rr["herd_iv"] = iv["n"][m, :, t].sum() / max(prep.n0[m].sum(), 1)
                    region.append(rr)
            safe = np.where(pop > 0, pop, np.nan)
            for i, c in enumerate(iso):
                for t in range(I.N_YEARS):
                    rr = dict(scenario=tg, bracket=bname, iso3=c, year=t + 1)
                    for k in COMPONENTS:
                        rr[k] = fs[k][i, t] / safe[i] / 365
                    rr.update(herd_ni=ni["n"][i, :, t].sum() / max(prep.n0[i].sum(), 1), herd_iv=iv["n"][i, :, t].sum() / max(prep.n0[i].sum(), 1),
                              reach_mu_iv=iv["access_mu"][i, t],
                              animal_ni=(ni["milk"][i, :, t].sum() + ni["meat"][i, :, t].sum()) / safe[i] / 365,
                              animal_iv=(iv["milk"][i, :, t].sum() + iv["meat"][i, :, t].sum()) / safe[i] / 365,
                              residue_produced_pj=prep.extras["e_res"][i, 0 if tg == 0 else t + 1, :].sum() / 1e9,
                              residue_feedable_pj=iv["fate"]["pool"][i, t] / 1e9 + iv["fate"]["stock_in"][i, t] / 1e9,
                              residue_fed_pj=iv["fate"]["fed"][i, t] / 1e9, residue_fed_without_pj=ni["res_used"][i, :, t].sum() / 1e9,
                              residue_extra_fed_pj=iv["fate"]["fed_extra"][i, t] / 1e9,
                              residue_unused_usual_pj=iv["fate"]["unused_usual"][i, t] / 1e9,
                              residue_carried_pj=iv["fate"]["carried"][i, t] / 1e9, residue_decayed_pj=iv["fate"]["decayed"][i, t] / 1e9)
                    country.append(rr)
                w = prep.e0[i]
                bind = iv["binding"][i, :, 1]
                dom = np.bincount(bind, weights=w, minlength=len(BINDING)).argmax() if w.sum() > 0 else 0
                binding.append(dict(scenario=tg, bracket=bname, iso3=c, dominant=BINDING[dom] if w.sum() > 0 else "no herd",
                                    **{hd: BINDING[bind[h]] if prep.n0[i, h] > 0 else "" for h, hd in enumerate(I.HERDS)}))
    I.RESULTS.mkdir(exist_ok=True)
    for name, rows in (("core_world", world), ("core_country", country), ("core_region", region), ("core_fate", fate),
                       ("core_herds", herds), ("core_binding", binding), ("core_residue_crops", crops)):
        pd.DataFrame(rows).round(4).to_csv(I.RESULTS / f"{name}.csv", index=False)
    w = pd.DataFrame(world)
    w["animal"] = w.milk + w.meat
    w["total"] = w.grain + w.animal + w.carcass
    pd.set_option("display.width", 220)
    print(w[["scenario", "bracket", "year", "grain", "animal", "carcass", "total", "herd_ni", "herd_iv"]].round(1).to_string())


if __name__ == "__main__":
    main()
