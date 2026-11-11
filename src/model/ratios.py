"""Residue-to-product ratios per country, crop and year, from the grain yield.

Every ratio source with a function of yield (residue_ratio_functions.csv)
is evaluated at the country's yield: the pre-catastrophe yield of crop_yield.csv,
scaled in each shock year by the change in production (harvested area is
held). The central ratio is the median of a crop's sources; its range is
the lowest and highest source at the same yield. A crop or member with a
single source takes the median relative spread of the crops with at least
three sources. A crop group is the production-weighted sum of its
members. Crops without functions keep the fixed ratio and range of
residue_feed_values.csv; a fixed ratio without a range there takes the same
median relative spread.

The functions were fitted to yields of ordinary farming. Yields are
therefore held within the 5th to 95th percentile of pre-catastrophe country
yields of each crop, weighted by harvested area (the yields at which the
crop is farmed today), so a collapsed harvest does not extrapolate the
ratio beyond the range of present-day yields. With
``hold=True`` the pre-catastrophe ratio of each country is kept in every year.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import inputs as I

BOUNDS = (0.05, 0.95)


def evaluate(form, p1, p2, y_fresh, dm):
    """Dry residue per dry product at fresh yield y_fresh (Mg per ha); dm is the product's dry-matter fraction."""
    y = np.asarray(y_fresh, dtype=float)
    if form == "ipcc":           # AGDM = Y_dm slope + intercept (IPCC 2006 Eq. 11.6, 11.7)
        return p1 + p2 / (y * dm)
    if form == "exponential":    # residue yield levels off above Y = 1/b (Bentsen et al. 2014; Smerald et al. 2023)
        return np.where(y <= 1 / p2, p1 * np.exp(-p2 * y), p1 / (p2 * np.e * y))
    if form == "log":
        return -p1 * np.log(y) + p2
    if form == "linear":
        return p1 - p2 * y
    if form == "constant":
        return np.full_like(y, p1)
    raise ValueError(form)


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return np.interp(q, cw, x[o])


@lru_cache(maxsize=None)
def _tables():
    crops = I.crop_tables(47).crops
    iso = I.countries()
    fn = I.read_data("residue_ratio_functions.csv")
    cy = I.read_data("crop_yield.csv")
    area = cy.pivot(index="iso3", columns="crop_code", values="area_ha").reindex(index=iso, columns=crops).fillna(0.0).to_numpy()
    bounds = {}
    for k in fn.crop_code.unique():
        sub = cy[(cy.crop_code == k) & (cy.area_ha > 0) & (cy.production_t > 0)]
        y, w = sub.yield_t_ha.to_numpy(float), sub.area_ha.to_numpy(float)
        world = sub.production_t.sum() / w.sum()
        bounds[k] = (*_weighted_quantile(y, w, np.array(BOUNDS)), world)
    return crops, iso, fn, area, bounds


def yields(tg: int, hold: bool = False) -> np.ndarray:
    """(C,K,T+1) fresh yield, Mg per ha, at held harvested area; NaN where there is no area."""
    crops, iso, fn, area, bounds = _tables()
    t = I.crop_tonnes(tg)
    if hold:
        t = np.repeat(t[:, :, :1], t.shape[2], axis=2)
    return np.where(area[:, :, None] > 0, t / np.where(area > 0, area, 1.0)[:, :, None], np.nan)


def central(tg: int, hold: bool = False) -> np.ndarray:
    """(C,K,T+1) central dry residue per dry product."""
    return _ratios(tg, hold)[0].copy()


def _spread_default():
    """Median (low, high) relative to the median of the crops with at least three sources, at world yield."""
    crops, iso, fn, area, bounds = _tables()
    lo, hi = [], []
    for k, sub in fn[fn.member == fn.crop_code].groupby("crop_code"):
        if len(sub) >= 3:
            v = np.array([evaluate(f.form, f.p1, f.p2, bounds[k][2], f.dm_product) for f in sub.itertuples()])
            lo.append(v.min() / np.median(v)); hi.append(v.max() / np.median(v))
    return float(np.median(lo)), float(np.median(hi))


@lru_cache(maxsize=None)
def _ratios(tg: int, hold: bool):
    """(central, low, high), each (C,K,T+1)."""
    crops, iso, fn, area, bounds = _tables()
    fv = I.read_data("residue_feed_values.csv").set_index("crop_code").reindex(crops)
    y = yields(tg, hold)
    shape = (len(iso), len(crops), y.shape[2])
    out = [np.broadcast_to(fv[c].to_numpy(float)[None, :, None], shape).copy() for c in ("rpr", "rpr_low", "rpr_high")]
    d_lo, d_hi = _spread_default()
    for o, d in ((out[1], d_lo), (out[2], d_hi)):   # a fixed ratio without a range of its own
        np.copyto(o, out[0] * d, where=np.isnan(o))
    for k, sub in fn.groupby("crop_code", sort=False):
        j = crops.index(k)
        lo, hi, world = bounds[k]
        yk = np.clip(np.nan_to_num(y[:, j, :], nan=world), lo, hi)
        acc = [np.zeros_like(yk) for _ in range(3)]
        for _, m in sub.groupby("member", sort=False):
            vals = np.maximum([evaluate(f.form, f.p1, f.p2, yk, f.dm_product) for f in m.itertuples()], 0.0)
            med = np.median(vals, axis=0)
            low, high = (vals.min(axis=0), vals.max(axis=0)) if len(m) > 1 else (med * d_lo, med * d_hi)
            w = m.weight.iloc[0]
            for a, v in zip(acc, (med, low, high)):
                a += w * v
        for o, a in zip(out, acc):
            o[:, j, :] = a
    return tuple(out)


def positioned(tg: int, pos, hold: bool = False) -> np.ndarray:
    """(C,K,T+1) ratio at a position in each crop's range (-1 low, 0 central, +1 high); pos scalar or (K,)."""
    c, lo, hi = _ratios(tg, hold)
    pos = np.clip(np.broadcast_to(np.asarray(pos, dtype=float), (c.shape[1],)), -1.0, 1.0)[None, :, None]
    return np.where(pos >= 0, c + pos * (hi - c), c + pos * (c - lo))
