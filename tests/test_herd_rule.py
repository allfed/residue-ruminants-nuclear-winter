"""The herd rule sensitivity (strategy "eaten"): free herds keep the most milk and meat eaten each year, every
animal kept beyond last year's herd costing its carcass now."""
import copy

import numpy as np
import pytest

from src.model.mechanism import (CONC, COP, GRASS, N_FEEDS, RES_SLICE, animal_output, simulate,
                                               solve_year)
from src.model.prepare import prepare

WITHIN = 0.005                         # stage 2 keeps the largest herd within 0.5% of the best (Batch.solve)


@pytest.fixture(scope="module")
def prep():
    p = prepare(47)
    q = copy.copy(p)
    q.n0 = np.where(np.isin(p.iso3, ["ETH", "AUS", "BRA"])[:, None], p.n0, 0.0)
    return q


def _avail(p, t, bracket):
    """Year t's usual feeds without the intervention, as simulate offers them."""
    a = np.zeros(p.n0.shape + (N_FEEDS,))
    conc = p.e0 * p.shares[:, :, 2]
    a[..., GRASS] = p.e0 * p.shares[:, :, 0] * p.grass_ratio[:, t - 1, None]
    a[..., CONC] = conc * p.conc_share * bracket
    a[..., COP] = conc * (1 - p.conc_share) * p.coproduct_ratio[:, t - 1, None]
    a[..., RES_SLICE] = p.res_use[:, :, t - 1, :] * p.extras["residue_me_multiplier"][:, t][:, None, :]
    return a


def _unchanged(p, ni, bracket, years=None):
    """(C,H,T) milk and meat eaten as chosen, and keeping last year's heads (the most output they can make);
    whether the herd grew, and whether last year's heads can be kept."""
    T = ni["n"].shape[2]
    chosen, alt, grows, ok = (np.zeros(ni["n"].shape) for _ in range(4))
    prev = p.n0.copy()
    for j in range(T):
        if years is None or j + 1 in years:
            r = solve_year(p, j + 1, prev, _avail(p, j + 1, bracket), n_fixed=prev, n_prev=prev)
            chosen[:, :, j] = ni["milk"][:, :, j] + ni["meat"][:, :, j]
            alt[:, :, j] = animal_output(p, r["n"], r["p"], prev)
            grows[:, :, j] = ni["n"][:, :, j] > prev * (1 + 1e-7)
            ok[:, :, j] = r["fixed_feasible"] & (prev > 0)
        prev = ni["n"][:, :, j]
    return chosen, alt, grows.astype(bool), ok.astype(bool)


@pytest.fixture(scope="module")
def runs(prep):
    return {b: simulate(prep, intervention=False, concentrate_ratio=b, strategy="eaten") for b in (1.0, 0.0)}


@pytest.mark.parametrize("bracket", [1.0, 0.0])
def test_growth_never_costs_food_this_year(prep, runs, bracket):
    """A herd grows only if that eats no less this year than keeping last year's heads, within stage 2's 0.5%."""
    chosen, alt, grows, ok = _unchanged(prep, runs[bracket], bracket)
    check = grows & ok
    assert check.sum() > 20                                              # herds do rebuild
    assert np.all(chosen[check] >= (1 - WITHIN - 1e-6) * alt[check])


def test_herd_rule_keeps_ethiopian_beef_from_rebuilding_at_a_loss(prep, runs):
    """Ethiopian beef, year 5, grain kept: the default rule grows the herd and eats far less than keeping last
    year's heads would; the herd rule eats as much."""
    c, h, j = prep.iso3.index("ETH"), prep.herds.index("beef_cattle"), 4
    default = simulate(prep, intervention=False, concentrate_ratio=1.0)
    chosen, alt, grows, ok = _unchanged(prep, default, 1.0, years={5})
    assert grows[c, h, j] and ok[c, h, j] and chosen[c, h, j] < 0.7 * alt[c, h, j]
    e_chosen, e_alt, _, e_ok = _unchanged(prep, runs[1.0], 1.0, years={5})
    assert e_ok[c, h, j] and e_chosen[c, h, j] >= (1 - WITHIN - 1e-6) * e_alt[c, h, j]
    assert e_chosen[c, h, j] > 1.5 * chosen[c, h, j]


def test_invalid_strategy_raises(prep):
    with pytest.raises(ValueError):
        simulate(prep, intervention=False, concentrate_ratio=1.0, strategy="eat")
    with pytest.raises(ValueError):
        solve_year(prep, 1, prep.n0, _avail(prep, 1, 1.0), strategy="most")
