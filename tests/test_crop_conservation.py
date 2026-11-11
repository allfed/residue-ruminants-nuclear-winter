"""Crop by crop, the usual and additional residue offered never exceed what the crop leaves in reach."""
import copy

import numpy as np
import pytest

from src.model.mechanism import RES_CROPS, access_share, crop_access_share, extra_share, simulate
from src.model.prepare import prepare


@pytest.fixture(scope="module")
def prep():
    return prepare(47)


def test_crop_pools_within_removable_production(prep):
    e, me_mult, T = prep.extras["e_res"], prep.extras["residue_me_multiplier"], prep.pool.shape[1]
    mu0 = prep.extras["eat_mu0"]
    reach = np.stack([access_share(prep, t, mu0[:, t]) for t in range(1, T + 1)], axis=1)   # (C,T)
    assert np.all(reach <= prep.tau[:, None] * (1 + 1e-3))                       # integer flow units round each cell's capacity
    crop_reach = np.stack([crop_access_share(prep, t, mu0[:, t]) for t in range(1, T + 1)], axis=1)   # (C,T,K) each crop alone
    assert np.all(crop_reach <= prep.tau[:, None, None] * (1 + 1e-3))
    within = e[:, 1:] * crop_reach * me_mult[:, 1:]                             # (C,T,K) in reach
    removable = e[:, 1:] * prep.tau[:, None, None] * me_mult[:, 1:]
    assert np.all(prep.pool_usual <= removable * (1 + 1e-6) + 1e-3)              # the rations' cap is a fixed point to 1e-7
    assert np.all(prep.pool <= np.maximum(within, prep.pool_usual) * (1 + 1e-9) + 1e-3)
    assert np.all(prep.pool <= np.maximum(within, removable) * (1 + 1e-6) + 1e-3)
    assert not np.any(prep.pool > removable * (1 + 1e-3) + 1e-3)                # no crop's pool above its removable residue
    # the onset stock: each crop's usual and additional shares stay within the reach of the pre-catastrophe harvest
    u, x = prep.crop_uptake, prep.extras["extra_share_crop"]
    a0 = crop_access_share(prep, 0, mu0[:, 0])
    live = np.isfinite(prep.uptake)
    assert np.all((u + x[:, 0])[live] <= np.maximum(a0, u)[live] * (1 + 1e-9) + 1e-12)
    assert np.all(x[~live] == 0) and np.all(prep.stock0[~live] == 0)
    sugc = RES_CROPS.index("SUGC")
    for iso in ("ARG", "MEX"):                                                   # their rations use about all the tops
        c = prep.iso3.index(iso)
        assert np.all(prep.pool[c, :, sugc] <= removable[c, :, sugc] * (1 + 1e-6))


def test_national_extra_is_reach_less_rations(prep):
    """The country is offered its reach less the rations' share, or less where its crops' own rooms (each crop's
    reach less its share in the rations) add up to less."""
    e, T = prep.extras["e_res"], prep.pool.shape[1]
    mu0 = prep.extras["eat_mu0"]
    produced = e.sum(axis=2)                                                     # (C,T+1)
    live = np.isfinite(prep.uptake)
    for t in range(T + 1):
        rations = prep.res_use[:, :, t - 1].sum(axis=(1, 2)) if t else np.where(live, prep.uptake, 0.0) * produced[:, 0]
        u_t = np.divide(rations, produced[:, t], out=np.zeros_like(rations), where=produced[:, t] > 0)
        budget = np.where(live, np.maximum(access_share(prep, t, mu0[:, t]) - u_t, 0.0), 0.0)
        room = np.maximum(crop_access_share(prep, t, mu0[:, t]) - prep.crop_uptake, 0.0)
        budget = np.minimum(budget, (room * prep.res_share[:, t]).sum(axis=1))
        offered = (extra_share(prep, t, mu0[:, t]) * e[:, t]).sum(axis=1)        # before decay and treatment
        assert np.allclose(offered, budget * produced[:, t], rtol=1e-9, atol=1e-3), t
        assert np.allclose(prep.extras["extra_share"][:, t], budget, rtol=1e-9, atol=1e-12), t
        if t:
            assert np.allclose(prep.pool[:, t - 1] - prep.pool_usual[:, t - 1],
                               prep.extra_base[:, t - 1] * extra_share(prep, t, mu0[:, t]), rtol=1e-9, atol=1e-3)
    assert np.allclose(prep.stock0, prep.stock_base * extra_share(prep, 0, mu0[:, 0]), rtol=1e-9, atol=1e-3)


def test_sugarcane_eaten_within_harvests(prep):
    """Argentina and Mexico, grain kept, no output lost: the sugarcane eaten over the years never exceeds the
    removable harvests plus the onset stock, nor the whole pre-catastrophe harvest."""
    p = copy.copy(prep)
    p.n0 = prep.n0.copy()
    p.n0[~np.isin(p.iso3, ["ARG", "MEX"])] = 0
    ni = simulate(p, intervention=False, concentrate_ratio=1.0)
    iv = simulate(p, intervention=True, concentrate_ratio=1.0, reference=ni, output_floor=0.0)
    k = RES_CROPS.index("SUGC")
    me = p.me[:, 0, :, 1 + k]                                                    # MJ/kg DM, not treated
    e = p.extras["e_res"][:, :, k]
    eaten = iv["diet_dm"][:, :, :, 1 + k].sum(axis=1).cumsum(axis=1)             # (C,T) kg DM
    harvests = (e[:, 1:] * p.tau[:, None] / me[:, 1:]).cumsum(axis=1)
    raw = harvests + (e[:, 0] * p.tau / me[:, 0])[:, None]
    tight = harvests + (p.stock0[:, k] / me[:, 1])[:, None]
    for iso in ("ARG", "MEX"):
        c = p.iso3.index(iso)
        assert eaten[c, -1] > 0
        assert np.all(eaten[c] <= raw[c] * (1 + 1e-6)), iso
        assert np.all(eaten[c] <= tight[c] * (1 + 1e-6)), iso
