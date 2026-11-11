"""Sensitivity: residue shared between countries, with no distance limit (Fig. 9c, d).

An estimate of what unlimited transport could add, in three intervention runs on one run without the intervention
(which offers no additional residue, so it is the same for all three):
  1. no distance limit within a country (theta "no_reach_limit"): each country offers all its removable residue;
  2. the residue never limits: every country is offered at least three times its herd's pre-catastrophe ME
     requirement a year, in its own crop mix of the year; what its herd then eats beyond run 1 is its unmet appetite;
  3. the world's spare residue of each crop and year in run 1 (eatable removable residue less what the herds ate
     of it, never negative) leaves its country and is shared among countries in proportion to their unmet appetite.
One pass: residue a country is sent but cannot eat is not passed on again.
"""
import numpy as np

from src.model.mechanism import food_saved, simulate
from src.model.prepare import prepare

FLOOR_TIMES_NEED = 3.0      # run 2: residue offered, in multiples of the herd's pre-catastrophe ME requirement


def run_shared(tg, theta=None, bracket=1.0, **switches):
    """Runs 1 to 3 at one scenario and bracket; returns (prep, ni, iv, saved) of run 3, as runs.core.run_pair."""
    prep = prepare(tg, {**(theta or {}), "no_reach_limit": True})
    sim_kw = {k: v for k, v in switches.items() if k in ("herd_dynamics", "strategy", "coproducts_gone", "tol", "scan")}
    ni = simulate(prep, intervention=False, concentrate_ratio=bracket, **sim_kw)

    def iv_run():
        return simulate(prep, intervention=True, concentrate_ratio=bracket, reference=ni, **switches)

    own = iv_run()
    prep.residue_floor = FLOOR_TIMES_NEED * prep.e0.sum(axis=1)[:, None, None] * prep.res_share[:, 1:]       # (C,T,K)
    unlimited = iv_run()
    prep.residue_floor = None
    eatable = (prep.tau[:, None, None] * prep.extras["e_res"][:, 1:] * prep.inyear
               * prep.extras["residue_me_multiplier"][:, 1:])                                            # (C,T,K)
    spare = np.maximum(eatable - own["fate"]["fed_by_crop"], 0.0)
    appetite = np.maximum(unlimited["fate"]["fed_extra"] - own["fate"]["fed_extra"], 0.0)                  # (C,T)
    weight = appetite / np.maximum(appetite.sum(axis=0, keepdims=True), 1e-30)
    prep.residue_out, prep.residue_in = spare, weight[:, :, None] * spare.sum(axis=0)[None]
    iv = iv_run()
    return prep, ni, iv, food_saved(prep, ni, iv)
