"""Reach at national levels: the residue allocated never exceeds what the animals can eat, whatever the draw,
and a crop out of reach offers nothing."""
import numpy as np
import pandas as pd
import pytest

from src.model.mechanism import (RES_CROPS, access_share, crop_access_share, demand_multiplier, extra_flow,
                                               extra_share, onset_stock)
from src.model.prepare import _access_table, _access_totals, _static, prepare

ROUNDING = 1e-4          # integer flow units: each cell's capacity is rounded by at most half a unit, about 1e-9 of the country total


@pytest.fixture(scope="module")
def preps():
    return {"central": prepare(47), "requirement 0.7": prepare(47, {"requirement_mult": 0.7})}


def capacity(p, n, t):
    """(C,) residue ME (MJ/yr) heads n can eat in year t: each head its NDF-limited residue-only intake, no more than its need."""
    per_head = np.where(p.n0 > 0, p.e0 / np.maximum(p.n0, 1e-9), 0.0)
    return (np.asarray(n) * per_head * p.eat_share[:, :, t]).sum(axis=1)


@pytest.mark.parametrize("case", ["central", "requirement 0.7"])
def test_allocated_residue_within_what_the_animals_can_eat(preps, case):
    p = preps[case]
    assert p.reach_scale is not None
    for k in (1.0, 0.4, 0.05):
        n = p.n0 * k
        for t in range(p.res_share.shape[1]):
            cap = capacity(p, n, t)
            mu = demand_multiplier(p, n, t)
            herd = p.e0.sum(axis=1) > 0
            assert np.allclose((mu * p.e0.sum(axis=1))[herd], cap[herd], rtol=1e-12)       # mu weights the species by their national requirement
            allocated = access_share(p, t, mu) * p.extras["e_res"][:, t].sum(axis=1)
            assert np.all(allocated <= cap * (1 + ROUNDING) + 1e-3), (case, k, t, p.iso3[int(np.argmax(allocated / np.maximum(cap, 1)))])
            alone = crop_access_share(p, t, mu) * p.extras["e_res"][:, t]                    # each crop alone, too
            assert np.all(alone <= cap[:, None] * (1 + ROUNDING) + 1e-3), (case, k, t)


def test_allocation_follows_what_the_animals_can_eat(preps):
    """Hungary, year 2: its residue far exceeds what its animals can eat, so the allocation is what they can eat,
    at central requirements and with requirements 30% lower, which lower it by more than 10%."""
    p, q = preps["central"], preps["requirement 0.7"]
    c, t = p.iso3.index("HUN"), 2
    caps = []
    for prep in (p, q):
        allocated = access_share(prep, t, prep.extras["eat_mu0"][:, t])[c] * prep.extras["e_res"][c, t].sum()
        caps.append(capacity(prep, prep.n0, t)[c])
        assert allocated == pytest.approx(caps[-1], rel=ROUNDING)
    assert caps[1] < 0.9 * caps[0]


def test_off_the_grid_no_reach(preps):
    p = preps["central"]
    residue, demand = _access_totals(_static(), 47)
    off = (residue[0] == 0) | (demand == 0)
    assert off.any()
    for t in (0, 3):
        assert np.all(access_share(p, t, np.ones(len(p.iso3)))[off] == 0)
        assert np.all(extra_share(p, t, np.ones(len(p.iso3)))[off] == 0)


@pytest.fixture
def crop_file(monkeypatch):
    """An in-memory per-crop file: every crop of every country takes its country's year-0 curve, except the wheat of
    the USA, which reaches no animal; the per-crop gridded residue splits the country's like the national residue."""
    s = _static()
    radii, ratios, tab, _ = _access_table(s, 47)
    residue, demand = _access_totals(s, 47)
    p = prepare(47)
    C, K = len(p.iso3), len(RES_CROPS)
    access = np.repeat(tab[0][:, None], K, axis=1)                                          # (C,K,R,ratio)
    usa, wheat = p.iso3.index("USA"), RES_CROPS.index("WHEA")
    access[usa, wheat] = 0.0
    z = dict(iso3=np.array(p.iso3), crops=np.array(RES_CROPS), radii=np.array(radii), ratios=ratios, access=access,
             residue_pj=residue[0][:, None] * p.res_share[:, 0] / 1e9, demand_pj=demand / 1e9, computed=np.ones(C, bool))
    monkeypatch.setitem(s, "access_crops", z)
    return usa, wheat


def test_a_crop_out_of_reach_offers_nothing(preps, crop_file):
    usa, wheat = crop_file
    p = prepare(47)
    assert p.crop_curve is not None
    mu0 = p.extras["eat_mu0"]
    assert p.extras["e_res"][usa, :, wheat].min() > 0 and p.extras["access"][usa, 0] > 0
    assert np.all(p.extras["extra_share_crop"][usa, :, wheat] == 0) and p.stock0[usa, wheat] == 0
    assert np.allclose(p.pool[usa, :, wheat], p.pool_usual[usa, :, wheat])
    for t in (0, 2, 10):
        # prepare and the mechanism agree
        assert np.allclose(crop_access_share(p, t, mu0[:, t]), p.extras["access_crop"][:, t] * p.access_mult, rtol=1e-12, atol=0)
        assert np.allclose(extra_share(p, t, mu0[:, t]), p.extras["extra_share_crop"][:, t], rtol=1e-12, atol=1e-15)
        for mu in (mu0[:, t], 0.3 * mu0[:, t]):
            x = extra_share(p, t, mu)
            assert x[usa, wheat] == 0
            # the country is offered its budget at most, and no crop more than its own reach
            w, u = p.res_share[:, t], p.crop_uptake
            live = np.isfinite(p.uptake)
            budget = np.maximum(access_share(p, t, mu) - np.multiply(u, w, out=np.zeros(w.shape), where=w > 0).sum(1), 0.0)
            assert np.all((x * w).sum(1)[live] <= budget[live] * (1 + 1e-12) + 1e-15)
            reach = crop_access_share(p, t, mu)
            assert np.all((u + x)[live] <= np.maximum(reach, u)[live] * (1 + 1e-12) + 1e-15)
    assert np.all(onset_stock(p, mu0[:, 0])[usa, wheat] == 0) and np.all(extra_flow(p, 4, mu0[:, 4])[usa, wheat] == 0)
    # at year 0 every other crop's curve and ratio are its country's: the other countries are offered as without the file
    s = _static()
    z, s["access_crops"] = s["access_crops"], None
    try:
        q = prepare(47)                                                    # truly without the per-crop file
    finally:
        s["access_crops"] = z
    others = np.arange(len(p.iso3)) != usa
    grown = p.extras["e_res"][:, 0] > 0
    assert np.allclose(p.extras["access"], q.extras["access"], rtol=0, atol=0)
    assert np.allclose(p.extras["access_crop"][:, 0][grown & others[:, None]], np.broadcast_to(q.extras["access"][:, :1], grown.shape)[grown & others[:, None]], rtol=1e-9)
    offered = lambda prep: prep.extras["extra_share_crop"][:, 0] * prep.res_share[:, 0]
    assert np.allclose(offered(p)[others], offered(q)[others], rtol=1e-9, atol=1e-15)


def test_without_the_per_crop_file_every_crop_takes_the_country_reach(preps):
    p = preps["central"]
    if p.crop_curve is not None:
        pytest.skip("the per-crop file is built")
    for t in (0, 5):
        mu = p.extras["eat_mu0"][:, t]
        assert np.array_equal(crop_access_share(p, t, mu), np.repeat(access_share(p, t, mu)[:, None], len(RES_CROPS), axis=1))


def test_the_reader_takes_either_ratio_grid(monkeypatch):
    """A table on the 31 ratios from 0.05 and one on the full grid down to 1.024e-4 are both read
    from their column names: low columns on the proportional chord, which is how the short grid is read below its first
    ratio, leave every share unchanged; a per-crop file on another grid than the table is refused."""
    from src.reach.access import RATIOS, SPLIT
    s = _static()
    acc = s["access"]
    radii = sorted({int(c.split("_")[1][:-2]) for c in acc.columns if c.startswith("access_")})
    have = sorted({float(c.split("_r")[1]) for c in acc.columns if c.startswith("access_")})
    short = acc[[c for c in acc.columns if not c.startswith("access_") or float(c.split("_r")[1]) >= SPLIT]]
    first = min(r for r in have if r >= SPLIT)
    low = {f"access_{R}km_r{r:g}": short[f"access_{R}km_r{first:g}"] * r / first for R in radii for r in RATIOS if r < SPLIT}
    extended = pd.concat([short, pd.DataFrame(low, index=short.index)], axis=1)
    monkeypatch.setitem(s, "access_crops", None)
    out = {}
    for name, table in (("short", short), ("extended", extended)):
        monkeypatch.setitem(s, "access", table)
        monkeypatch.setitem(s, "access_tab", {})
        monkeypatch.setitem(s, "access_totals", {})
        out[name] = prepare(47)
    assert len(out["short"].access_ratios) == 31 and np.array_equal(out["extended"].access_ratios, RATIOS)
    assert np.allclose(out["extended"].extras["access"], out["short"].extras["access"], rtol=1e-12, atol=1e-15)
    for t in (0, 4):
        for k in (1.0, 0.01, 0.001):
            mu = out["short"].extras["eat_mu0"][:, t] * k
            assert np.allclose(access_share(out["extended"], t, mu), access_share(out["short"], t, mu), rtol=1e-12, atol=1e-15)
    z = dict(iso3=np.array(out["short"].iso3), crops=np.array(RES_CROPS), radii=np.array(radii), ratios=np.array(RATIOS),
             access=np.zeros((len(out["short"].iso3), len(RES_CROPS), len(radii), len(RATIOS))),
             residue_pj=np.ones((len(out["short"].iso3), len(RES_CROPS))), demand_pj=np.ones(len(out["short"].iso3)),
             computed=np.ones(len(out["short"].iso3), bool))
    monkeypatch.setitem(s, "access", short)
    monkeypatch.setitem(s, "access_tab", {})
    monkeypatch.setitem(s, "access_crops", z)
    with pytest.raises(ValueError, match="different ratio grids"):
        prepare(47)
