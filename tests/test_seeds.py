"""Fixed herds start from the ration eaten without the intervention: that seed is checked, and every
ration returned meets energy, MP and NDF whatever the solver said."""
import numpy as np
import pytest

from src.model import protein, ration
from src.model.mechanism import (CONC, COP, GRASS, N_FEEDS, RES_SLICE, animal_output, efficiencies,
                                               solve_year)
from src.model.prepare import prepare


def _pair(prep, t):
    """Year t without the intervention, then the same heads and output with the intervention, no extra residue;
    and each seed's violation of its problem."""
    avail = np.zeros(prep.n0.shape + (N_FEEDS,))
    avail[..., GRASS] = prep.e0 * prep.shares[:, :, 0] * prep.grass_ratio[:, t - 1, None]
    avail[..., CONC] = prep.e0 * prep.shares[:, :, 2] * prep.conc_share
    avail[..., COP] = prep.e0 * prep.shares[:, :, 2] * (1 - prep.conc_share) * prep.coproduct_ratio[:, t - 1, None]
    avail[..., RES_SLICE] = prep.res_use[:, :, t - 1, :]
    ni = solve_year(prep, t, prep.n0, avail)
    ref = {k: ni[k] for k in ("dm", "md", "p", "urea")}
    ref["animal"] = animal_output(prep, ni["n"], ni["p"], prep.n0)
    seen, check = [], ration.Batch.violation
    ration.Batch.violation = lambda self, *a, **kw: seen.append(check(self, *a, **kw)) or seen[-1]
    try:
        iv = solve_year(prep, t, prep.n0, avail, extra=np.zeros_like(avail[..., RES_SLICE]), n_fixed=ni["n"],
                        reference_year=ref, displace=True)
    finally:
        ration.Batch.violation = check
    return ni, iv, np.array(seen)


@pytest.fixture(scope="module")
def mix():
    prep = prepare(47, {"usual_residue": "mix"})
    return (prep, 1) + _pair(prep, 1)


def test_mixture_seed_meets_energy(mix):
    """The usual residue mixture is seeded in kg DM: every reference ration is a feasible seed, and no fixed
    herd loses the energy of its residue."""
    prep, t, ni, iv, seen = mix
    assert len(seen) == (ni["n"] > 0).sum() and np.all(seen <= ration.SEED_TOL)
    assert np.array_equal(iv["n"], ni["n"]) and np.allclose(iv["p"], ni["p"], rtol=0, atol=1e-6)
    assert iv["fixed_feasible"].all() and iv["nutrients_ok"].all()
    e, dm = iv["used"].sum(-1), iv["dm"].sum(-1)
    km, kl, kg = efficiencies(e / np.maximum(dm, 1e-20))
    need = iv["n"] * 365 * (prep.ne_m * prep.cold[:, :, t] / km + iv["p"] * (prep.ne_l / kl + prep.ne_g / kg + prep.ne_p / km))
    assert np.all(e + 1e-6 * np.maximum(need, 1) >= need)


def test_infeasible_seed_is_dropped():
    """A seed that misses the energy row is never the answer: the search finds a feasible ration, or none."""
    base = dict(me=np.array([[9.0, 7.0, 13.0, 12.0]]), ndf=np.array([[0.6, 0.75, 0.15, 0.3]]), mp=np.full((1, 2, 4), 0.2),
                mp_urea=np.zeros(2), nb=np.full((1, 4), 0.03), lb=np.zeros((1, 4)), tier=np.array([[3.0, 4.0, 1.0, 2.0]]),
                treated=np.zeros((1, 4)), pairs=[], grain=2, g0=np.array([10.0]), nem=np.array([40.0]),
                nel=np.array([20.0]), neg=np.zeros(1), nep=np.zeros(1), mpm=np.array([0.01]), mpp=np.array([0.01]),
                cap=np.array([20.0]), lw=np.array([500.0]), pcap=np.ones(1), free=False, floor=np.ones(1), objective="grain",
                outk=np.ones(1), gk=np.array([3000.0]))
    seed = (np.array([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]), np.array([9.0]))       # no feed, full production: cheaper than any ration
    bt = ration.Batch(ub=np.array([[100.0, 100.0, 10.0, 0.0]]), **base)
    x, v, q, ok, _ = bt.search_one(0)
    assert ok and bt.violation(0, x, q) <= ration.SEED_TOL                   # a solver ration passes the check
    x0 = np.concatenate([seed[0][0], [0.0, 0.0]])
    assert bt.violation(0, x0, 9.0) > 0.1
    r = bt.solve(anchor=seed[1], seed=seed)
    assert r["feasible"][0] and r["nutrients_ok"][0] and r["p"][0] == 1.0 and r["feeds"][0].sum() > 1.0
    none = ration.Batch(ub=np.zeros((1, 4)), **base).solve(anchor=seed[1], seed=seed)   # nothing to eat
    assert not none["feasible"][0]


def _small_ruminant(ub, free=True, p=1.0):
    """One sheep-like problem fed grass, a residue, grain and co-products (per reference head and day)."""
    return ration.Batch(me=np.array([[9.08, 7.56, 10.96, 11.67]]), ndf=np.array([[0.6, 0.75, 0.15, 0.3]]),
                        mp=np.full((1, 2, 4), 0.08), mp_urea=np.zeros(2), nb=np.full((1, 4), 0.03), lb=np.zeros((1, 4)),
                        ub=ub, tier=np.array([[3.0, 4.0, 1.0, 2.0]]), treated=np.zeros((1, 4)), pairs=[], grain=2,
                        g0=np.array([0.037]), nem=np.array([5.35]), nel=np.zeros(1),
                        neg=np.array([0.18]), nep=np.array([0.24]), mpm=np.array([0.047]), mpp=np.array([0.0074]),
                        cap=np.array([0.56]), lw=np.array([40.0]), pcap=np.array([p]), free=free, floor=np.array([p]),
                        objective="output" if free else "grain", outk=np.ones(1), gk=np.array([3000.0]))


@pytest.mark.parametrize("scale", [1e-5, 1e-6])
def test_tiny_herd_is_solved_at_its_scale(scale):
    """A free herd that keeps a tiny share of its heads (Kazakhstan's and Tajikistan's sheep and goats in year 2 at
    150 Tg) gets the ration and production per head kept of the same herd solved at its own scale, not one that
    misses its rows or bounds by the solver's absolute tolerance; and that ration seeds the fixed herd of the
    intervention, which keeps the same heads and production."""
    offer = np.array([[0.822, 5.4e-4, 0.037, 0.026]])
    big = _small_ruminant(offer * 0.1).solve()
    r = _small_ruminant(offer * scale).solve()
    assert np.isclose(r["nu"][0] / scale, big["nu"][0] / 0.1, rtol=1e-9, atol=0)
    assert np.isclose(r["p"][0], big["p"][0], rtol=1e-9, atol=0) and r["nutrients_ok"][0]
    nu, p = r["nu"][0], r["p"][0]
    ref = r["feeds"] / nu                                                # what each head kept ate
    fixed = _small_ruminant(ref * (1 + 1e-9) + 1e-12, free=False, p=p)
    seed = (np.concatenate([ref[0], [1.0, p]])[None], r["md"])
    out = fixed.solve(anchor=r["md"], seed=seed, within=None)
    assert out["feasible"][0] and out["nutrients_ok"][0] and out["p"][0] >= p - 1e-6
