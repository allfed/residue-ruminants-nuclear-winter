"""Nitrogen accounting and the energy ceiling on microbial protein."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.model import protein
from src.model.mechanism import K_RES, N_FEEDS
from src.runs.urea_budget import budget


def test_urea_closes_only_the_nitrogen_gap():
    p = SimpleNamespace(n0=np.ones((1, 1)), protein_model="mp", urea_max_dm=.01,
        cp=np.array([[[[.1]]]]), rdp=np.array([[[[.02]]]]), drup=np.array([[[[.03]]]]),
        tdn=np.array([[[[.5]]]]), treatment_npn=np.zeros((1, 1, 1, 1)),
        mp_m=np.array([[.05]]), mp_p=np.array([[.02]]), extras={"live_weight": np.array([[500.]])})
    x = np.ones((1, 1, 1))
    urea = protein.supplement(p, 0, x, np.array([[.5]]))
    supplied = .64/1.18*(.02+protein.UREA_CP*urea)+.03
    assert supplied == pytest.approx(.06)
    coeff, offset = protein.coefficients(p, 0)
    supply_caps = np.einsum("chlf,chf->chl", coeff, x)+offset
    assert supply_caps[0, 0, 0] == pytest.approx(.64*.13*.5+.03)
    # More nitrogen cannot exceed the independent microbial-energy bound.
    p.urea_max_dm = 0
    no_urea, _ = protein.coefficients(p, 0)
    assert no_urea[0, 0, 0, 0] == coeff[0, 0, 0, 0]


def test_budget_counts_only_material_actually_treated():
    straw = np.zeros((1, 2, 1, K_RES))
    straw[0, 0, 0, 1:3] = [.5, 1.]                     # kg DM of crops 1 and 2 eaten treated by the first herd
    r = {"treated_dm": straw, "feed_treatment_npn": np.zeros((1, 2, 2, N_FEEDS)), "urea": np.zeros((1, 2, 1))}
    r["feed_treatment_npn"][0, :, 1, 2:4] = [.02, .04]  # both herds' feed values; only the first ate it treated
    b = budget(None, r, 0)
    assert b["straw_treated_mt_dm"] == pytest.approx(1.5e-9)
    assert b["treatment_n_retained_mt"] == pytest.approx((.5 * .02 + 1 * .04) / 6.25 / 1e9)
    assert b["treatment_n_retained_mt"]+b["treatment_n_lost_mt"] == pytest.approx(b["urea_treatment_mt"]*.46)
