"""Urea budget of the cereal-straw treatment process.

Treatment has unit dry-matter yield, a stated process approximation. Added
urea DM leaves in the aggregate processing loss; retained treatment N is
included in the output composition and subtracted from the N loss. Only
straw actually eaten treated receives treatment: a herd eats eligible straw
untreated where treating it would not improve its ration (least urea). This is not a farm dosing guide.
"""
import pandas as pd

from src import inputs as I
from src.model.mechanism import RES_SLICE
from src.model.prepare import prepare
from src.runs.core import BRACKETS, run_pair

WORLD_UREA_MT = 199.7
LEVER = {"treatment": True}
APPLICATION = .05


def budget(prep, result, t):
    straw = result["treated_dm"][:, :, t]                                        # (C,H,K) kg DM eaten treated
    treated_dm = straw.sum() / 1e9
    retained_n = (straw * result["feed_treatment_npn"][:, :, t+1][..., RES_SLICE]).sum() / 6.25 / 1e9
    applied = treated_dm * APPLICATION
    return dict(straw_treated_mt_dm=treated_dm, urea_treatment_mt=applied,
                urea_treatment_low=treated_dm*.04, urea_treatment_high=treated_dm*.06,
                treatment_n_retained_mt=retained_n, treatment_n_lost_mt=applied*.46-retained_n,
                treatment_process_dm_loss_mt=applied, treatment_dm_yield=1.0,
                share_of_world_urea=applied/WORLD_UREA_MT)


def main():
    rows = []
    for tg in I.SCENARIOS:
        prep = prepare(tg, LEVER)
        for bname, b in BRACKETS.items():
            _, _, iv, _ = run_pair(tg, bracket=b, prep_iv=prep)
            for t in range(I.N_YEARS):
                rows.append(dict(scenario=tg, bracket=bname, year=t+1, **budget(prep, iv, t)))
    df = pd.DataFrame(rows)
    df.to_csv(I.RESULTS / "urea_budget.csv", index=False)
    print(df.groupby(["scenario", "bracket"])[["straw_treated_mt_dm", "urea_treatment_mt", "share_of_world_urea"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
