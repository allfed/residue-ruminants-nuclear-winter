"""From the built data files and a parameter set to the arrays of the mechanism.

Each counted crop's residue is its own feed (mechanism.RES_CROPS) with the
crop's positioned ME, CP, NDF, RDP and digestible RUP; straw treatment is
applied per crop, country and year. Field decay follows each crop's decay
type and the monthly temperature where the crop is grown (decay_windows). The residue fill order (res_order) ranks the crops of each country and
year by ME per kg NDF of the feed offered, ties by higher ME; the ration
takes usual and extra residue in that order (theta usual_residue="ranked").
The alternative "mix" feeds the usual residue as a fixed mixture in the
proportions offered; extra residue stays ranked.
"""

from __future__ import annotations

from types import SimpleNamespace
from functools import lru_cache

import numpy as np
import pandas as pd

from src import inputs as I
from src.model import params as P
from src.model import protein, ratios
from src.model.mechanism import (CONC, COP, GE_FEED, GRASS, K_RES, N_FEEDS, RES_CROPS, RES_SLICE,
                                               Prepared, crop_extra_share, crop_reach_share, demand_multiplier, efficiencies,
                                               residue_order)

DAYS_PER_MONTH = 365.25 / 12
MAY = 4                  # the war starts in May: model year t runs from May of weather year t to April of year t + 1
LARGE = np.array([True, True, True, True, False, False])
QUANT = {"rpr": "rpr_mult", "me": "residue_me_mult", "cp": "residue_cp_mult", "ndf": "residue_ndf_mult"}
TREATED_N_DEGRADED = 2.0 / 3.0   # share of the N retained by treated straw that the rumen microbes use (Chenost and Kayouli 1997)


@lru_cache(maxsize=None)
def _static():
    iso = I.countries()
    crops = I.crop_tables(47).crops
    fv = I.read_data("residue_feed_values.csv").set_index("crop_code").reindex(crops)
    herds = I.read_data("ruminant_herds.csv")
    herds["region"] = herds["iso3"].map(I.regions())
    h = {c: herds.pivot(index="iso3", columns="herd", values=c).reindex(index=iso, columns=list(I.HERDS)).to_numpy(float)
         for c in herds.columns if c not in ("iso3", "region", "herd")}
    sp = I.read_data("stream_properties.csv").set_index(["region", "herd"])
    reg = [I.regions()[c] for c in iso]
    stream = {c: np.array([[sp.loc[(r, hd), c] for hd in I.HERDS] for r in reg]) for c in sp.columns if c != "ration_from"}
    reserve = I.read_data("residue_reserve.csv").set_index("region").reindex(reg)
    access = I.read_data("residue_access.csv") if (I.DATA / "residue_access.csv").exists() else None
    access_crops = None                                     # each crop alone (residue_access_crops.npz); without it every crop takes the country's reach
    if (I.DATA / "residue_access_crops.npz").exists():
        with np.load(I.DATA / "residue_access_crops.npz") as z:
            access_crops = {k: z[k] for k in z.files}
        assert list(access_crops["iso3"]) == list(iso) and tuple(access_crops["crops"]) == tuple(RES_CROPS) and access_crops["computed"].all()
    ps = I.read_data("protein_streams.csv").set_index(["region", "herd"])
    prot_stream = {c: np.array([[ps.loc[(r, hd), c] for hd in I.HERDS] for r in reg]) for c in ps.columns}
    # grass and hay: temperate and tropical values mixed by the share of each herd's heads in temperate climate (grass_climate.csv)
    sh = I.read_data("grass_climate.csv").pivot(index="iso3", columns="herd", values="temperate_share").reindex(
        index=iso, columns=list(I.HERDS)).to_numpy(float)
    for q in ("me", "cp", "ndf"):
        for suffix in ("", "_low", "_high"):
            stream[f"grass_{q}{suffix}"] = sh * stream[f"grass_temperate_{q}{suffix}"] + (1 - sh) * stream[f"grass_tropical_{q}{suffix}"]
    w_t, w_tr = sh * stream["grass_temperate_cp"], (1 - sh) * stream["grass_tropical_cp"]      # protein fractions weighted by protein
    for c in ("rdp_fraction", "digestible_rup_fraction"):
        prot_stream[f"grass_{c}"] = (w_t * prot_stream[f"grass_temperate_{c}"] + w_tr * prot_stream[f"grass_tropical_{c}"]) / (w_t + w_tr)
    prot_crop = I.read_data("protein_residues.csv").set_index("crop_code").reindex(crops)
    cal = I.read_data("residue_calendar.csv")
    months = cal.set_index(["iso3", "crop_code"])[[f"m{m}" for m in range(1, 13)]]
    calendar = np.stack([months.loc[c].reindex(crops).to_numpy(float) for c in iso])        # (C,K,12)
    cold = dict(np.load(I.DATA / "cold_deficit.npz"))                              # loaded, not an open file: safe across forks
    decay_temp = dict(np.load(I.DATA / "decay_temperature.npz"))
    assert list(decay_temp["iso3"]) == list(iso) and tuple(decay_temp["crops"]) == tuple(RES_CROPS)
    # share of each harvest month's residue whose crop is not yet sown again, x months after harvest (residue_clearing.npz);
    # crops without a planting map are not sown every year and keep their residue
    cl = np.load(I.DATA / "residue_clearing.npz")
    assert list(cl["iso3"]) == list(iso)
    unsown = np.ones((len(iso), len(RES_CROPS), 12, len(cl["months"])))
    for j, code in enumerate(cl["crops"]):
        unsown[:, RES_CROPS.index(code)] = cl["unsown"][:, j]
    clearing = dict(months=cl["months"], unsown=unsown)
    base = I.herd_products(herds)           # meat moved to the dairy herd if the meat herd is empty or above its carcass ceiling
    is_milk = base["product"].isin([p for ps in I.MILK_OF.values() for p in ps])
    milk0 = np.zeros((len(iso), len(I.HERDS)))
    meat0 = np.zeros_like(milk0)
    for j, hd in enumerate(I.HERDS):
        for out, sel in ((milk0, is_milk), (meat0, ~is_milk)):
            out[:, j] = base[sel & (base.herd == hd)].groupby("iso3")["tcal"].sum().reindex(iso).fillna(0.0).to_numpy() * I.KCAL_PER_TCAL
    counted = np.array([crops.index(c) for c in RES_CROPS])                     # positions of the residue feeds
    # RUSLE2 decomposition coefficient of each counted crop, per month at the optimum (residue_decay_crops.csv)
    phi = I.read_data("residue_decay_crops.csv").set_index("crop_code")["phi_per_day"].reindex(list(RES_CROPS))
    assert not phi.isna().any(), phi[phi.isna()].index.tolist()
    phi = phi.to_numpy(float) * DAYS_PER_MONTH
    # national bound on the rations' grain and bran (grain_feed.csv): kg DM per year, none where the PSD has no row
    gb = I.read_data("grain_feed.csv").set_index("iso3")["bound_dm_t"].reindex(iso)
    grain_bound = np.where(gb.notna(), gb.to_numpy(float) * 1000.0, np.inf)
    return dict(iso=iso, crops=crops, fv=fv, h=h, stream=stream, reserve=reserve, access=access, access_crops=access_crops, calendar=calendar,
                clearing=clearing, cold=cold, decay_temp=decay_temp, milk0=milk0, meat0=meat0, counted=counted, phi=phi, reg=reg, prot_stream=prot_stream, prot_crop=prot_crop,
                grain_bound=grain_bound)


def _positioned(fv, q, pos):
    return _between(np.asarray(fv[q], float), np.asarray(fv[q + "_low"], float), np.asarray(fv[q + "_high"], float), pos)


def _between(mode, lo, hi, pos):
    """Position -1 is the low end of the range, 0 the central value, +1 the high end."""
    pos = np.clip(pos, -1.0, 1.0)
    return np.where(pos >= 0, mode + pos * (hi - mode), mode + pos * (mode - lo))


SPECIES_OF_HERD = {"dairy_cattle": 0, "beef_cattle": 0, "dairy_buffaloes": 1, "meat_buffaloes": 1,
                   "dairy_small_ruminants": 2, "meat_small_ruminants": 2}      # cattle, buffalo, small ruminants (sheep and goats)


def _access_table(s, tg):
    """The reach table (residue_access.csv) for scenario tg: (radii, ratios, tab (T+1, C, radius, ratio), species (C, 3)).

    tab is g(r), the share of the country's residue that the maximum flow allocates with the
    whole residue removable and the pre-catastrophe demand scaled by r; year 0 is pre-catastrophe.
    species is the gridded pre-catastrophe demand of cattle, buffaloes and small ruminants (PJ), a
    diagnostic since the demand multiplier weights the species by their national requirement.
    The radii and ratios are read from the column names, access_{R}km_r{r}."""
    cache = s.setdefault("access_tab", {})
    if tg not in cache:
        acc = s["access"]
        access_cols = [c for c in acc.columns if c.startswith("access_")]
        radii = sorted({int(c.split("_")[1][:-2]) for c in access_cols})
        ratios = sorted({float(c.split("_r")[1]) for c in access_cols})
        col = lambda r, k: f"access_{r}km_r{ratios[k]:g}"
        sub = acc[acc.scenario == tg]
        if sub.empty:
            raise ValueError(f"residue_access.csv has no scenario {tg}")
        years = [sub[sub.year == y].set_index("iso3").reindex(s["iso"]) for y in range(I.N_YEARS + 1)]
        tab = np.stack([np.stack([np.stack([a[col(r, k)].to_numpy(float) for k in range(len(ratios))], axis=1)
                                  for r in radii], axis=1) for a in years])
        y0 = years[0]
        if all(f"demand_{sp}_pj" in y0.columns for sp in ("cattle", "buffalo", "sheep", "goats")):
            species = np.stack([y0["demand_cattle_pj"].to_numpy(float), y0["demand_buffalo_pj"].to_numpy(float),
                                y0["demand_sheep_pj"].to_numpy(float) + y0["demand_goats_pj"].to_numpy(float)], axis=1)
        else:
            species = None
        cache[tg] = (radii, np.asarray(ratios, float), np.nan_to_num(tab), species)
    return cache[tg]


def _access_totals(s, tg):
    """The table's gridded totals for scenario tg, the levels of the grid's patterns: residue (T+1, C) and
    demand (C,), MJ per year; None for a table without them (the model then reads g at mu / tau)."""
    cache = s.setdefault("access_totals", {})
    if tg not in cache:
        acc = s["access"]
        if not {"year", "residue_pj", "demand_pj"} <= set(acc.columns):
            cache[tg] = None
        else:
            sub = acc[acc.scenario == tg]
            years = [sub[sub.year == y].set_index("iso3").reindex(s["iso"]) for y in range(I.N_YEARS + 1)]
            cache[tg] = (np.nan_to_num(np.stack([a["residue_pj"].to_numpy(float) for a in years])) * 1e9,
                         np.nan_to_num(years[0]["demand_pj"].to_numpy(float)) * 1e9)
    return cache[tg]


def reach_scale(cap0, supply, grid_supply, grid_demand):
    """The table ratio per unit of mu, cap0 grid_supply / (supply grid_demand) (broadcast), zero where any of them
    is zero. cap0 is the national pre-catastrophe requirement (mu cap0 is what the herd can eat, Cap) and supply the
    national residue S; the grid gives the patterns. The flow is homogeneous in its capacities, so the share of S
    that the flow of S and Cap spread on those patterns moves is tau g(mu s / tau), never more than Cap / S."""
    cap0, supply, grid_supply, grid_demand = np.broadcast_arrays(*(np.asarray(a, float) for a in (cap0, supply, grid_supply, grid_demand)))
    ok = (cap0 > 0) & (supply > 0) & (grid_supply > 0) & (grid_demand > 0)
    return np.where(ok, cap0 * grid_supply / np.where(ok, supply * grid_demand, 1.0), 0.0)


def left_on_field(theta, rv):
    """(C,) share of the residue left on the field (residue_reserve.csv): what harvest leaves uncut and animals do not graze,
    by region; intense_collection takes the most thorough region's value everywhere."""
    return rv["soil_low" if theta.get("intense_collection") else "soil"].to_numpy(float)


def access_curve(theta, s, tg):
    """(T+1, C, ratio) g at the draw's radius, interpolated in log radius between the built radii,
    and the ratio grid; None without a table or with no distance limit (then the whole removable residue is in reach)."""
    if s["access"] is None or theta.get("no_reach_limit"):                                   # sensitivity: no distance limit within a country
        return None, None
    radii, ratios, tab, _ = _access_table(s, tg)                                             # (T+1,C,R,ratio)
    i, w = _radius_weight(theta, radii)
    return (1 - w) * tab[:, :, i, :] + w * tab[:, :, i + 1, :], ratios


def _radius_weight(theta, radii):
    """The built radius i below the draw's radius and its weight w against radius i + 1, linear in log radius."""
    x = np.log(np.clip(theta["access_radius_km"], radii[0], radii[-1]))
    lr = np.log(radii)
    i = int(np.clip(np.searchsorted(lr, x) - 1, 0, len(radii) - 2))
    return i, (x - lr[i]) / (lr[i + 1] - lr[i])


def crop_curve(theta, s):
    """(C, K, ratio) g_k of each crop alone (residue_access_crops.npz, year-0 grid) at the draw's radius, interpolated in
    log radius as access_curve, and the ratio grid; None without the per-crop file."""
    z = s["access_crops"]
    if z is None:
        return None, None
    i, w = _radius_weight(theta, np.asarray(z["radii"], float))
    tab = np.nan_to_num(z["access"])                                                        # (C,K,R,ratio)
    return (1 - w) * tab[:, :, i, :] + w * tab[:, :, i + 1, :], np.asarray(z["ratios"], float)


def share_at(curve, ratios, tau, mu):
    """(C,) allocated share of the residue, tau g(mu / tau), from one year's curve (C, ratio): linear in the
    ratio (a mixture of two feasible flows is feasible), proportional below the first ratio (a scaled-down
    flow) and constant above the last (the flow of the last ratio). Zero demand or supply gives zero."""
    tau = np.clip(np.asarray(tau, float), 0.0, 1.0)
    mu = np.clip(np.asarray(mu, float), 0.0, None)
    if curve is None:
        return tau * (mu > 0)
    r = np.divide(mu, tau, out=np.zeros_like(tau), where=tau > 0)
    grid = np.concatenate([[0.0], ratios])
    values = np.concatenate([np.zeros((curve.shape[0], 1)), curve], axis=1)
    g = np.array([np.interp(rc, grid, row) for rc, row in zip(r, values)])
    return tau * np.clip(g, 0.0, 1.0)                                         # integer flow units can round past one


def access_share(theta, s, tau, tg, mu=None):
    """(C, T+1) allocated share of each year's production at table ratio mu (C, T+1; the demand multiplier times
    reach_scale; default 1, the whole gridded demand): reach (radius) and the share allowed to leave the field (tau)."""
    curve, ratios = access_curve(theta, s, tg)
    if curve is None:
        return np.repeat(np.asarray(tau, dtype=float)[:, None], I.N_YEARS + 1, axis=1)
    mu = np.ones((len(tau), I.N_YEARS + 1)) if mu is None else mu
    return np.stack([share_at(curve[t], ratios, tau, mu[:, t]) for t in range(I.N_YEARS + 1)], axis=1)


def eat_share(ndf_cap, me_head, me, ndf, res_share, me_plain=None):
    """(C,H,T+1) residue ME a head can eat in a day, an NDF-limited diet of nothing but the country's residue
    mix of the year (ndf_cap over the mix's NDF per MJ), over its requirement me_head, at most one. me and ndf
    (C,H,T+1,F) carry any treatment; res_share (C,T+1,K) is the residue ME split by crop. With me_plain (the
    untreated values), the share is in residue ME as produced, the unit of the reach's supply: the ME eaten
    divided by the treatment's uplift of the mix."""
    per_mj = (res_share[:, None] * ndf[..., RES_SLICE] / np.maximum(me[..., RES_SLICE], 1e-9)).sum(-1)   # kg NDF per MJ of the mix
    can = np.divide(ndf_cap[..., None], per_mj, out=np.full(per_mj.shape, np.inf), where=per_mj > 0)
    share = np.clip(np.divide(can, me_head[..., None], out=np.ones(per_mj.shape), where=me_head[..., None] > 0), 0.0, 1.0)
    if me_plain is not None:
        w = res_share[:, None].sum(-1)
        uplift = (res_share[:, None] * me[..., RES_SLICE] / np.maximum(me_plain[..., RES_SLICE], 1e-9)).sum(-1)   # ME eaten per MJ produced
        uplift = np.divide(uplift, w, out=np.ones_like(uplift), where=w > 0)          # no residue: nothing to scale
        share = np.clip(share / np.where(uplift > 0, uplift, 1.0), 0.0, 1.0)
    return share


def cold_multiplier(tg, theta, s):
    """(C,H,T+1) maintenance multiplier from cold; year 0 is the pre-catastrophe climate.
    1 + acclim_rate x the drop of the annual mean temperature from the pre-catastrophe climate, plus c x the
    mean monthly deficit below L of each class, with L and c those of P.COLD_LCT and P.COLD_COEF moved
    together by cold_wet toward the wet, windy case."""
    z = s["cold"]
    si = list(z["scenarios"]).index(tg)
    d = z["deficit"][si]                                       # (C, group, class, year, L, exposure)
    out = np.ones((d.shape[0], len(I.HERDS), d.shape[3]))
    if theta.get("no_cold_stress"):
        return out
    grid = z["lct_grid"]
    wet = theta["cold_wet"]
    for g, name in enumerate(("large", "small")):
        direct = np.zeros((d.shape[0], d.shape[3], 2))
        for k in range(2):                                     # class 0: coldest month below 5 C; class 1: warmer
            lct = np.clip(P.COLD_LCT[name][k] + wet * P.COLD_WET_SHIFT, grid[0], grid[-1])
            coef = P.COLD_COEF[name][k] * (1 + wet * (P.COLD_WET_FACTOR - 1))
            j = np.clip(np.searchsorted(grid, lct) - 1, 0, len(grid) - 2)
            w = (lct - grid[j]) / (grid[j + 1] - grid[j])
            direct += coef * ((1 - w) * d[:, g, k, :, j, :] + w * d[:, g, k, :, j + 1, :])
        direct = (1 - theta["shelter"]) * direct[..., 0] + theta["shelter"] * direct[..., 1]
        acclim = theta["acclim_rate"] * z["cooling"][si][:, g, :]
        out[:, LARGE == (g == 0), :] = (1.0 + acclim + direct)[:, None, :]
    return out


def unsown_share(theta, s, ages=None, onset=True):
    """Share of each harvest month's residue whose crop is not yet sown again (residue_clearing.npz), ages months from the
    middle of the harvest month, by default the start of the month one to twelve months after the harvest
    month: (C,K,harvest month,age). Residue nobody uses is gone at the next sowing; clearing_shift moves that
    sowing later (+) or earlier (-), in months. no_clearing (no sowing before onset) concerns only the stock
    at onset: onset=False ignores it."""
    tab = s["clearing"]["unsown"]
    ages = np.arange(12) + 0.5 if ages is None else np.asarray(ages, float)
    if onset and theta.get("no_clearing"):
        return np.ones(tab.shape[:3] + ages.shape)
    g = s["clearing"]["months"]
    x = ages - theta["clearing_shift"]                                            # months from mid-harvest, less the shift
    pos = np.clip(x / (g[1] - g[0]), 0, len(g) - 1)
    i0 = np.minimum(pos.astype(int), len(g) - 2)
    f = pos - i0
    return np.where(x < 0, 1.0, tab[..., i0] * (1 - f) + tab[..., i0 + 1] * f)


def rusle2_tf(t, t_opt=32.0, a=8.0):
    """RUSLE2 temperature function of residue decomposition, equation 10.6: one at the optimum
    temperature, zero below -10 C (USDA-ARS 2013, p. 250). Applied to monthly mean air temperature."""
    x, xo = np.asarray(t, float) + a, t_opt + a
    tf = (2 * x ** 2 * xo ** 2 - x ** 4) / xo ** 4
    return np.where(np.asarray(t) < -10.0, 0.0, np.clip(tf, 0.0, 1.0))


def decay_windows(theta, s, tg, shock=True, parts=False):
    """Monthly decay of the additional residue, counted crops (temperatures of decay_temperature.npz).

    The residue of a harvest is fed over the twelve months from the middle of
    its harvest month, in twelve equal parts of what was harvested, one taken
    at the end of each month and diminished by the decay since harvest. What
    is not eaten is left at the end of the twelve months and is fed over the
    next twelve in the same way, in its form: the baled or stacked part still
    decays at k_stored, the field part at the field rates. Field residue
    decays at each month's rate (the crop's RUSLE2 coefficient times the
    RUSLE2 temperature function, without its moisture function), baled or
    stacked residue at k_stored. Field residue still standing when its crop
    is sown again is gone (residue_clearing.npz), harvested after onset or standing at
    onset; crops without a planting map are not cleared. Returns

      inyear (C,T+1,K)  share of a year's residue that can be eaten,
      carry  (C,T+1,K)  residue left after the twelve months per unit of that
                        eatable share left uneaten (one without decay or sowing),
      rem, w_on, r_on (onsets,C,K)  for each onset month, the share of the
                        last twelve months' harvests standing at onset (crop
                        not yet sown again, field decay, pre-catastrophe
                        climate) and the eatable and remaining shares of it
                        over the twelve months from onset, its field part
                        gone when its crop is sown (no_clearing: never),
      and with parts a dict, by year fed (C,T+1,K):
        carry_stored    the baled or stacked part of carry,
        inyear_carried  (...,2) share of the residue carried into the year,
                        baled or stacked and field, that can be eaten,
        keep_carried    (...,2) share of it left at the end of the year.
    """
    z = s["decay_temp"]
    tm = z["tmon"][list(z["scenarios"]).index(tg)].astype(float)                  # (C,K,11,12), calendar months
    if not shock:
        tm = np.repeat(tm[:, :, :1], tm.shape[2], axis=2)
    T = tm.shape[2] - 1
    rate = lambda t: theta["field_decay_factor"] * s["phi"][None, :, None] * rusle2_tf(t)
    # months May to April; year 0 is the pre-catastrophe climate, year y takes January to April from weather year y + 1
    years = [np.concatenate([tm[:, :, 0, MAY:], tm[:, :, 0, :MAY]], axis=2)]
    years += [np.concatenate([tm[:, :, min(y, T), MAY:], tm[:, :, min(y + 1, T), :MAY]], axis=2) for y in range(1, T + 3)]
    kf = rate(np.concatenate(years, axis=2))                                      # (C,K,months)
    cum = np.concatenate([np.zeros(kf.shape[:2] + (1,)), np.cumsum(kf, axis=2)], axis=2)   # loss to the start of each month
    mid = cum[..., :-1] + 0.5 * kf
    ks, b = theta["k_stored"], theta["baled_share"]
    j = np.arange(12)

    start = np.arange(12 * (T + 1))                                               # harvests of years 0..T
    # field residue is gone at its crop's next sowing: each twelfth takes the share unsown at its age, the rest that at
    # twelve months; what is left then is fed the next year at ages 12.5 to 23.5, the rest left at 24 months
    u = unsown_share(theta, s, np.r_[np.arange(24) + 0.5, 12.0, 24.0], onset=False)[:, :, (start + MAY) % 12]   # (C,K,harvest,26)
    surv = lambda i: np.exp(-(cum[..., start + i + 1] - mid[..., start]))          # field: harvest to the end of month i after it
    wf = np.mean([surv(i) * u[..., i] for i in j], axis=0)
    rf = np.exp(-(mid[..., start + 12] - mid[..., start])) * u[..., 24]
    ws, rs = np.exp(-ks * (j + 0.5)).mean(), np.exp(-12 * ks)                     # baled or stacked
    w = b * ws + (1 - b) * wf
    r = b * rs + (1 - b) * rf
    cal = np.nan_to_num(s["calendar"][:, s["counted"]])                           # (C,K,12)
    cal = np.concatenate([cal[..., MAY:], cal[..., :MAY]], axis=2)
    tot = cal.sum(axis=2, keepdims=True)
    cal = np.where(tot > 0, cal / np.where(tot > 0, tot, 1.0), 1 / 12)
    by_year = lambda x: np.einsum("ckyp,ckp->cyk", x.reshape(x.shape[:2] + (T + 1, 12)), cal)
    inyear, keep = by_year(w), by_year(r)
    carry = np.where(inyear > 0, keep / np.maximum(inyear, 1e-12), 1.0)
    extra = None
    if parts:
        # the field part left at twelve months (crops not yet sown again) is fed the next year at that year's
        # rates until its crop is sown; per unit of it, zero where there is none (the crop was sown)
        fc = by_year(rf)
        cond = lambda x: np.where(fc > 0, by_year(x) / np.where(fc > 0, fc, 1.0), 0.0)
        w2 = cond(np.mean([surv(12 + i) * u[..., 12 + i] for i in j], axis=0))
        r2 = cond(np.exp(-(mid[..., start + 24] - mid[..., start])) * u[..., 25])
        nxt = lambda x: np.concatenate([x[:, :1], x[:, :-1]], axis=1)             # harvest of year y, fed in year y + 1
        one = np.ones_like(inyear)
        extra = dict(carry_stored=np.where(inyear > 0, b * rs / np.maximum(inyear, 1e-12), carry),
                     inyear_carried=np.stack([ws * one, nxt(w2)], axis=-1), keep_carried=np.stack([rs * one, nxt(r2)], axis=-1))

    onset = theta["onset"]
    onsets = range(12) if onset in ("uniform", "worst") else [int(onset) - 1]
    cal_m = np.nan_to_num(s["calendar"][:, s["counted"]])                         # calendar months again
    k0 = rate(tm[:, :, 0, :])                                                     # (C,K,12) pre-catastrophe
    # share unsown at ages 0.5 to 23.5 months: at onset (a - 0.5 for the harvest a months before), when the
    # twelfth i after onset is taken (a + 0.5 + i) and at the end of the twelve months (a + 11.5)
    later = unsown_share(theta, s, np.arange(24) + 0.5)                           # (C,K,harvest month,age)
    a = np.arange(1, 13)
    rem, w_on, r_on = [], [], []
    for m0 in onsets:
        back = (m0 - a) % 12                                                      # harvested one to twelve months before
        kb = k0[..., back]
        loss = np.cumsum(kb, axis=2) - 0.5 * kb                                    # from the middle of the harvest month
        coh = cal_m[..., back] * np.exp(-loss)                                    # (C,K,a)
        lb = later[:, :, back]                                                    # (C,K,a,age)
        standing = (coh * lb[:, :, a - 1, a - 1]).sum(axis=2)
        rem.append(standing)
        s0 = 12 + (m0 - MAY) % 12                                                 # the onset month of year 1
        dec = np.stack([np.exp(-(cum[..., s0 + i + 1] - cum[..., s0])) for i in j], axis=-1)   # (C,K,12) field decay from onset
        rf_on = np.exp(-(cum[..., s0 + 12] - cum[..., s0]))
        # the field part is gone when its crop is sown after onset: unsown at a + 0.5 + i over unsown at onset
        fw = (coh * (lb[:, :, (a - 1)[:, None], a[:, None] + j] * dec[:, :, None, :]).mean(axis=3)).sum(axis=2)
        fr = (coh * lb[:, :, a - 1, a + 11]).sum(axis=2) * rf_on
        ok = standing > 0
        w_on.append(b * np.exp(-ks * (j + 1)).mean() + (1 - b) * np.where(ok, fw / np.where(ok, standing, 1.0), dec.mean(axis=2)))
        r_on.append(b * rs + (1 - b) * np.where(ok, fr / np.where(ok, standing, 1.0), rf_on))
    out = inyear, carry, np.stack(rem), np.stack(w_on), np.stack(r_on)
    return out + (extra,) if parts else out


def residue_energy(tg, theta, s):
    """Residue ME (MJ), DM (kg), CP and NDF (kg) per country, crop, year 0..10."""
    fv = s["fv"]
    val = {q: _positioned(fv, q, theta[p]) for q, p in QUANT.items() if q != "rpr"}
    # the ratio follows each country's yield (model/ratios.py); rpr_hold_baseline keeps the pre-catastrophe ratio in every year
    val["rpr"] = ratios.positioned(tg, theta["rpr_mult"], hold=bool(theta.get("rpr_hold_baseline")))
    dm = I.crop_tonnes(tg) * 1000.0 * fv["dm_product"].to_numpy(float)[None, :, None] * val["rpr"]
    return dm, val


def prepare(tg: int, theta: dict | None = None, *, shock: bool = True) -> Prepared:
    theta = {**P.defaults(), **(theta or {})}
    s = _static()
    iso, h, st = s["iso"], s["h"], s["stream"]
    C, H, T = len(iso), len(I.HERDS), I.N_YEARS
    req = theta["requirement_mult"]

    # ---- residue production by crop
    dm, val = residue_energy(tg, theta, s)                                     # (C,K,T+1)
    hold = lambda a, axis: np.repeat(np.take(a, [0], axis=axis), T + 1, axis=axis)
    if not shock:
        dm = hold(dm, 2)
    # Treatment is applied per crop to eligible cereal materials in warm country-years.
    eligible = np.isin(s["fv"].index, ("WHEA", "RICE", "MAIZ", "BARL", "SORG", "MILL", "PMIL", "OCER"))
    treated = np.zeros_like(dm, dtype=bool)
    if theta.get("treatment"):
        z = s["cold"]
        temp = z["tmean"][list(z["scenarios"]).index(tg)].mean(axis=1)
        if not shock:
            temp = hold(temp, 1)
        warm = temp >= theta.get("treatment_min_temp", -np.inf)
        treated = eligible[None, :, None] & warm[:, None, :]
        treated[:, :, 0] = False
    tr = {**P.TREATMENT, **(theta.get("treatment_values") or {})}
    crop_me = val["me"][None, :, None] + treated * tr["me_add"]
    crop_cp = val["cp"][None, :, None] / 100 + treated * tr["cp_add"] / 100
    crop_ndf = val["ndf"][None, :, None] / 100 / np.where(treated, tr["ndf_intake_gain"], 1)
    r = s["prot_crop"]["rdp_fraction"].to_numpy(float)[None, :, None]
    d = s["prot_crop"]["rup_digestibility"].to_numpy(float)[None, :, None]
    # of the nitrogen the treated straw retains, 30 to 35% is bound to indigestible cell walls and unused by the
    # rumen microbes (Chenost and Kayouli 1997, ch. 5): two thirds count as RDP, the rest is neither RDP nor RUP
    crop_rdp = val["cp"][None, :, None] / 100 * r + treated * tr["cp_add"] / 100 * TREATED_N_DEGRADED
    crop_drup = val["cp"][None, :, None] / 100 * (1-r) * d
    crop_npn = treated * tr["cp_add"] / 100
    cnt = s["counted"]
    by_crop = lambda arr: np.broadcast_to(arr, dm.shape)[:, cnt, :].transpose(0, 2, 1)   # (C,K_all,T+1) to (C,T+1,K)
    e_res = by_crop(dm * val["me"][None, :, None])                             # untreated ME produced per crop
    e_tot = e_res.sum(axis=2)
    res_ratio = np.where(e_tot[:, :1] > 0, e_tot[:, 1:] / np.maximum(e_tot[:, :1], 1e-9), 1.0)
    res_share = np.where(e_tot[..., None] > 0, e_res / np.maximum(e_tot[..., None], 1e-9), 1.0 / K_RES)
    me = np.zeros((C, H, T + 1, N_FEEDS)); cp = np.zeros_like(me); ndf = np.zeros_like(me)
    rdp, drup, npn = np.zeros_like(me), np.zeros_like(me), np.zeros_like(me)
    for stream, f, key in (("grass", GRASS, "grass"), ("concentrates", CONC, "conc"), ("coproducts", COP, "coproduct")):
        value = lambda q: _between(st[f"{stream}_{q}"], st[f"{stream}_{q}_low"], st[f"{stream}_{q}_high"], theta[f"{key}_{q}_pos"])
        me[..., f] = value("me")[:, :, None]
        cp[..., f] = value("cp")[:, :, None] / 100
        ndf[..., f] = np.maximum(value("ndf"), 1.0)[:, :, None] / 100
        rdp[..., f] = cp[..., f] * s["prot_stream"][stream + "_rdp_fraction"][:, :, None]
        drup[..., f] = cp[..., f] * s["prot_stream"][stream + "_digestible_rup_fraction"][:, :, None]
    for arr, values in ((me, crop_me), (cp, crop_cp), (ndf, crop_ndf), (rdp, crop_rdp), (drup, crop_drup), (npn, crop_npn)):
        arr[..., RES_SLICE] = by_crop(values)[:, None]                           # feed values are per crop, same for all herds
    untreated_me = me.copy()
    untreated_me[..., RES_SLICE] = by_crop(val["me"][None, :, None])[:, None]
    res_order = residue_order(me[:, 0, :, RES_SLICE], ndf[:, 0, :, RES_SLICE])  # (C,T+1,K)

    # ---- requirements and the pre-catastrophe diet
    cold = cold_multiplier(tg, theta, s)
    if not shock:
        cold = hold(cold, 2)
    ne_m, ne_l, ne_g, ne_p = (h[k] * req for k in ("ne_m", "ne_l", "ne_g", "ne_p"))
    n0 = h["heads"]
    # ---- the pre-catastrophe ration
    # Mottet et al. (2017) give regional rations as dry-matter shares (stream_properties.csv, dm_*); this draw's feed values turn them into
    # energy. Named sugarcane tops are the residue of sugarcane (SUGC); the other crop residues follow the country's
    # residue production, net of the tops. The rations are regional averages, whose roughages in developing countries
    # follow the roughages available in each cell (GLEAM 2.0 model description, 3.2.3). Where a country's herds would
    # eat more tops than its cane leaves, the rest of the tops ration becomes other crop residue; where they would eat
    # more crop residue than may leave its fields (production less the shares left on the field and taken by other
    # uses, at their central values: the pre-catastrophe reserve, the same in every draw), the excess moves to grass
    # and other roughages, in every herd of the country alike. Likewise, where its herds would eat more grain and bran
    # than the country's cereal feed for all livestock plus the bran of the wheat and rice it eats as food (grain_feed.csv), the
    # excess moves to grass and other roughages. Intake follows diet quality, so all are iterated to a fixed point.
    me0 = me[:, :, 0, :]
    me_crop0 = me0[:, 0, RES_SLICE]                                                   # (C,K) the same for every herd
    k_sugc = RES_CROPS.index("SUGC")
    e_prod = e_res[:, 0, :]                                                           # (C,K) residue ME produced
    rv0 = s["reserve"]
    tau0 = 1.0 - np.clip(rv0["soil"].to_numpy(float) + rv0["other"].to_numpy(float), 0.0, 0.95)   # (C,) may leave the field
    e_free = e_prod * tau0[:, None]                                                   # (C,K) what the rations may draw on
    e_sugc = e_free[:, k_sugc]
    dm_tot = sum(st[f"dm_{k}"] for k in ("grass", "crop_residues", "sugarcane_tops", "concentrates", "coproducts"))
    dmr = {k: st[f"dm_{k}"] / dm_tot for k in ("grass", "crop_residues", "sugarcane_tops", "concentrates", "coproducts")}   # rounding
    e_grass, e_conc, e_cop = dmr["grass"] * me0[..., GRASS], dmr["concentrates"] * me0[..., CONC], dmr["coproducts"] * me0[..., COP]
    grain_bound = s["grain_bound"]                                                    # (C,) kg DM per year
    tops_full = dmr["sugarcane_tops"] * me_crop0[:, None, k_sugc]                  # (C,H) energy per kg of ration DM
    onehot = np.zeros(K_RES)
    onehot[k_sugc] = 1.0
    no_cap = bool(theta.get("no_residue_cap"))                                        # switch: the rations as published
    tops_f = np.where(e_sugc > 0, 1.0, 0.0)                                           # share of the tops ration kept as tops
    gen_f = np.ones(C)                                                                # share of the other crop residue kept
    grain_f = np.ones(C)                                                              # share of the grain and bran ration kept
    tops_use = np.zeros(C)
    for _ in range(30):
        avail = np.maximum(e_free - tops_use[:, None] * onehot, 0.0)                  # what the other residues may draw on
        mix = np.where(avail.sum(1, keepdims=True) > 0, avail / np.maximum(avail.sum(1, keepdims=True), 1e-12),
                       np.where(e_prod.sum(1, keepdims=True) > 0, e_prod / np.maximum(e_prod.sum(1, keepdims=True), 1e-12), 1.0 / K_RES))
        me_gen = 1.0 / (mix / me_crop0).sum(1)                                        # (C,) ME per kg DM of the other residues
        e_tops = tops_full * tops_f[:, None]
        gen_full = (dmr["crop_residues"] + dmr["sugarcane_tops"] * (1 - tops_f[:, None])) * me_gen[:, None]
        e_gen = gen_full * gen_f[:, None]
        e_grain = e_conc * grain_f[:, None]
        e_all = e_grass + gen_full * (1 - gen_f[:, None]) + e_tops + e_gen + e_conc + e_cop
        shares = np.stack([e_grass + gen_full * (1 - gen_f[:, None]) + e_conc - e_grain, e_tops + e_gen, e_grain + e_cop], -1) / e_all[..., None]
        split = np.where(e_grain + e_cop > 0, e_grain / np.maximum(e_grain + e_cop, 1e-12), st["concentrate_me_share"])
        comp = (e_tops[..., None] * onehot + e_gen[..., None] * mix[:, None, :]) / np.maximum(e_tops + e_gen, 1e-12)[..., None]
        comp = np.where((e_tops + e_gen)[..., None] > 0, comp, mix[:, None, :])        # (C,H,K) the ration's residue by crop
        share_conc, share_cop = shares[:, :, 2] * split, shares[:, :, 2] * (1 - split)
        inv_me = shares[:, :, 0] / me0[..., GRASS] + share_conc / me0[..., CONC] + share_cop / me0[..., COP] + \
            shares[:, :, 1] * (comp / me0[..., RES_SLICE]).sum(-1)
        q0 = 1.0 / inv_me
        km, kl, kg = efficiencies(q0, theta.get("efficiency_system", "ipcc"))
        me_head = ne_m * cold[:, :, 0] / km + ne_l / kl + ne_g / kg + ne_p / km         # MJ ME per head per day
        e0 = n0 * me_head * 365.0
        tops_claim = (e0 * tops_full / e_all).sum(1)                                  # MJ/yr the rations claim as tops
        gen_claim = (e0 * gen_full / e_all).sum(1)
        claimed = (e0 * (tops_full + dmr["crop_residues"] * me_gen[:, None]) / e_all).sum(1)
        grain_claim = (e0 * e_conc / e_all / me0[..., CONC]).sum(1)                   # kg DM/yr of the full grain and bran ration
        if no_cap:
            break
        new_tops = np.where(tops_claim > e_sugc, e_sugc / np.maximum(tops_claim, 1e-9), 1.0) * (e_sugc > 0)
        tops_use = tops_claim * new_tops
        room = np.maximum(e_free.sum(1) - tops_use, 0.0)
        new_gen = np.where(gen_claim > room, room / np.maximum(gen_claim, 1e-9), 1.0)
        new_grain = np.where(grain_claim > grain_bound, grain_bound / np.maximum(grain_claim, 1e-9), 1.0)
        if np.allclose(new_tops, tops_f, rtol=0, atol=1e-7) and np.allclose(new_gen, gen_f, rtol=0, atol=1e-7) \
                and np.allclose(new_grain, grain_f, rtol=0, atol=1e-7):
            break
        tops_f, gen_f, grain_f = new_tops, new_gen, new_grain
    produced0 = e_prod.sum(1)
    cap_f = gen_f
    dm_head = me_head * inv_me
    frac = lambda arr: shares[:, :, 0] / me[:, :, 0, GRASS] * arr[:, :, 0, GRASS] + share_conc / me[:, :, 0, CONC] * arr[:, :, 0, CONC] + \
        share_cop / me[:, :, 0, COP] * arr[:, :, 0, COP] + \
        shares[:, :, 1] * (comp / me[:, :, 0, RES_SLICE] * arr[:, :, 0, RES_SLICE]).sum(-1)
    ndf_head0 = me_head * frac(ndf)
    cp_head0 = me_head * frac(cp)
    cap = np.where(LARGE[None, :], theta["ndf_cap_large"], theta["ndf_cap_small"]) / 100.0 * h["lw_kg"]
    # Every herd's capacity is the larger of the ceiling and its pre-catastrophe NDF intake times ndf_baseline_factor
    # (a judgement, 1.00 to 1.20): no herd is held below the ceiling, nor below its own intake plus that margin. Herds
    # above the ceiling and herds within the margin below it (e.g. Latin American beef) both take the anchored value.
    ndf_floor = ndf_head0 * theta["ndf_baseline_factor"] > cap
    ndf_cap = np.maximum(cap, ndf_head0 * theta["ndf_baseline_factor"])
    if theta.get("no_intake_ceiling"):
        ndf_cap = ndf_cap * 100.0
    # MP requirements are built directly, without a dietary CP conversion.
    raw_mp_m = h["mp_m"] * theta["protein_mult"]
    raw_mp_p = h["mp_p"] * theta["protein_mult"]
    model = theta.get("protein_model", "mp")
    if model not in ("mp", "cp_proxy"):
        raise ValueError("protein_model must be mp or cp_proxy")
    # ME/DE=0.82 and DE=4.409 Mcal/kg TDN are an explicit approximate
    # conversion, not the full NRC dairy energy model. Hold the NRC intake
    # discount at this draw's pre-catastrophe diet so the MP constraints stay linear.
    tdn = me / (0.82 * 4.409 * 4.184)
    tdn0 = me_head * frac(tdn) / np.maximum(dm_head, 1e-9)
    multiples = me_head / np.maximum(ne_m * cold[:, :, 0] / km, 1e-9)
    discounted = np.maximum(0.60, tdn0 - np.maximum(0.18 * tdn0 - 0.103, 0) * np.maximum(multiples - 1, 0))
    discount = np.where(tdn0 > 0.60, np.minimum(discounted / tdn0, 1), 1)
    tdn *= discount[:, :, None, None]
    # A herd may eat treated straw untreated: keep the untreated values of each residue.
    untreated = None
    if treated.any():
        res_shape = (C, H, T + 1, K_RES)
        plain = lambda values: np.broadcast_to(by_crop(np.broadcast_to(values, dm.shape))[:, None], res_shape).copy()
        u_me = plain(val["me"][None, :, None])
        untreated = dict(me=u_me, cp=plain(val["cp"][None, :, None] / 100), ndf=plain(val["ndf"][None, :, None] / 100),
                         rdp=plain(val["cp"][None, :, None] / 100 * r), drup=plain(val["cp"][None, :, None] / 100 * (1 - r) * d),
                         tdn=u_me / (0.82 * 4.409 * 4.184) * discount[:, :, None, None], npn=np.zeros(res_shape))
    mp_energy = me_head * frac(protein.MICROBIAL_MP * protein.MICROBIAL_CP_PER_TDN * tdn + drup)
    mp_nitrogen = me_head * frac(protein.MICROBIAL_MP / protein.RDP_PER_MICROBIAL_CP * rdp + drup)
    mp_head0 = np.minimum(mp_energy, mp_nitrogen) if model == "mp" else protein.MICROBIAL_MP * cp_head0
    # Where the pre-catastrophe ration supplies less MP than the requirement, both parts of the requirement are scaled
    # by one factor, keeping their ratio; herds with enough MP keep the raw requirement. The factor is fixed after the shock.
    raw_mp = raw_mp_m + raw_mp_p
    mp_alpha = np.where(raw_mp > 0, np.minimum(1.0, mp_head0 * (1 - 1e-6) / np.maximum(raw_mp, 1e-12)), 1.0)
    mp_m = mp_alpha * raw_mp_m
    mp_p = np.maximum(mp_alpha * raw_mp_p, 1e-6)                                     # the mechanism divides by mp_p
    cp_short = mp_alpha < 1.0
    if theta.get("no_protein_constraint"):
        mp_m, mp_p = mp_m * 0.0, np.full_like(mp_p, 1e-9)
    urea_max_dm = float(theta.get("urea_max_dm", 0))
    if not 0 <= urea_max_dm <= 0.01:
        raise ValueError("Supplemental urea cap must be between zero and 1% of diet DM")

    # ---- uptake, reserve, access, pool
    base_use = e0[..., None] * shares[:, :, 1, None] * comp                          # (C,H,K) the rations' residue ME by crop
    base_crop_res = base_use.sum(axis=(1, 2))
    uptake = np.where(e_tot[:, 0] > 0, base_crop_res / np.maximum(e_tot[:, 0], 1e-9), np.inf)
    # each crop's own share in the rations, the same every year as the rations follow each crop's production
    use_k = base_use.sum(axis=1)                                                     # (C,K)
    crop_uptake = np.divide(use_k, e_res[:, 0], out=np.where(use_k > 0, np.inf, 0.0), where=e_res[:, 0] > 0)
    crop_uptake = np.where(np.isfinite(uptake)[:, None], crop_uptake, np.inf)        # nothing produced: nothing offered
    rv = s["reserve"]
    reserve = left_on_field(theta, rv) + rv["other"].to_numpy(float) * theta["reserve_other_mult"]
    if theta.get("bedding_reserve"):                                                 # sensitivity: bedding kept from the feed
        reserve = reserve + np.where(rv.index.isin(P.BEDDING_REGIONS), P.BEDDING_SHARE, 0.0)
    reserve = np.clip(reserve, 0.0, 0.95)
    tau = 1.0 - reserve
    curve, ratios = access_curve(theta, s, tg)                                       # (T+1,C,ratio) g at the draw's radius
    if curve is not None and not shock:
        curve = np.repeat(curve[:1], T + 1, axis=0)
    access_mult = float(theta.get("access_mult", 1.0))
    # the residue the animals can eat: each head's NDF-limited residue-only intake, no more than its need
    eat = eat_share(ndf_cap, me_head, me, ndf, res_share, me_plain=untreated_me)    # (C,H,T+1), in residue ME as produced
    if not shock:
        eat = hold(eat, 2)
    herd_species = np.array([SPECIES_OF_HERD[hd] for hd in I.HERDS])
    # species weighted by their national pre-catastrophe requirement in this draw: mu times the national requirement
    # is the residue ME the heads can eat (Cap), whatever the grid's species mix
    species_weight = np.stack([np.where(herd_species == k, e0, 0.0).sum(axis=1) for k in range(3)], axis=1)
    herds0 = SimpleNamespace(n0=n0, e0=e0, eat_share=eat, species_weight=species_weight, herd_species=herd_species)
    mu0 = np.stack([demand_multiplier(herds0, n0, t) for t in range(T + 1)], axis=1)            # (C,T+1) the whole herd
    # national levels on the grid's patterns: the table is read at mu s, s = E0 E_grid / (S D_grid) (reach_scale)
    scale = None
    totals = _access_totals(s, tg) if curve is not None else None
    if totals is not None:
        e_grid, d_grid = totals                                                      # (T+1,C), (C,) MJ
        if not shock:
            e_grid = np.repeat(e_grid[:1], T + 1, axis=0)
        scale = reach_scale(e0.sum(axis=1)[:, None], e_tot, e_grid.T, d_grid[:, None])   # (C,T+1)
    access = access_share(theta, s, tau, tg, mu0 if scale is None else mu0 * scale)   # (C,T+1) share at the pre-catastrophe herd
    if not shock:
        access = hold(access, 1)
    # each crop alone (residue_access_crops.npz, year-0 grid): its pattern moves by one national factor, so its national residue
    # of the year S_k and Cap on the year-0 patterns give the ratio mu s_k, s_k = E0 E_grid_k(0) / (S_k D_grid)
    crops_g = crop_curve(theta, s)[0] if scale is not None else None
    crop_scale = crop_access = None
    if crops_g is not None:
        z = s["access_crops"]
        if not np.array_equal(np.asarray(z["ratios"], float), ratios):                # the mechanism reads both on prep.access_ratios
            raise ValueError(f"residue_access_crops.npz ({len(z['ratios'])} ratios) and residue_access.csv ({len(ratios)}) have different "
                             "ratio grids; both must be tabulated on src/reach/access.RATIOS")
        crop_scale = reach_scale(e0.sum(axis=1)[:, None, None], e_res, z["residue_pj"][:, None, :] * 1e9, z["demand_pj"][:, None, None] * 1e9)
        crop_access = np.stack([crop_reach_share(crops_g, ratios, tau, mu0[:, t, None] * crop_scale[:, t]) for t in range(T + 1)], axis=1)
    extra_crop = np.stack([crop_extra_share(access[:, t] * access_mult, crop_uptake, res_share[:, t],
                                            None if crop_access is None else crop_access[:, t] * access_mult)
                           for t in range(T + 1)], axis=1)                           # (C,T+1,K), mu = 1
    extra_share = (extra_crop * res_share).sum(axis=2)                               # (C,T+1) the country's budget, less where the crops' rooms fall short

    z = s["cold"]
    inyear, carry, rem, w_on, r_on, carried_parts = decay_windows(theta, s, tg, shock, parts=True)   # (C,T+1,K), (onsets,C,K)
    me_mult = me[:, 0, :, RES_SLICE] / np.maximum(untreated_me[:, 0, :, RES_SLICE], 1e-9)   # (C,T+1,K) treatment uplift
    crop_ratio = np.where(e_res[:, :1] > 0, e_res[:, 1:] / np.maximum(e_res[:, :1], 1e-12), 1.0)   # (C,T,K) each crop's own
    res_use = base_use[:, :, None, :] * crop_ratio[:, None]                          # (C,H,T,K) the rations' residue, untreated
    pool_usual = res_use.sum(axis=1) * me_mult[:, 1:]                                # (C,T,K) the rations' residue
    extra_base = e_res[:, 1:] * inyear[:, 1:] * me_mult[:, 1:]                       # (C,T,K) extra residue per unit of extra share
    pool = pool_usual + extra_base * extra_crop[:, 1:]                               # (C,T,K) at the pre-catastrophe herd

    # ---- stock at onset: the unused residue of the last twelve months' harvests whose crop is not yet sown again
    e_crop0 = (dm[:, :, 0] * val["me"][None, :])[:, cnt]                             # (C,K)
    standing = e_crop0[None] * rem                                                   # (onsets,C,K) at onset
    eatable, left = standing * w_on, standing * r_on                                 # over the twelve months from onset
    onset_month = None
    if theta["onset"] == "worst":                    # the one start month, for every country, that leaves the world the smallest stock
        world = (eatable * me_mult[None, :, 1] * extra_crop[None, :, 0]).sum(axis=(1, 2))
        o = int(world.argmin())
        onset_month = o + 1
        eatable, left = eatable[o], left[o]
    else:
        eatable, left = eatable.mean(axis=0), left.mean(axis=0)
    carry_onset = np.where(eatable > 0, left / np.maximum(eatable, 1e-12), 1.0)      # (C,K)
    # its baled or stacked part; the rest is the field part of crops not sown again
    standing_o = standing[onset_month - 1] if onset_month else standing.mean(axis=0)
    carry_onset_stored = np.where(eatable > 0, standing_o * theta["baled_share"] * np.exp(-12 * theta["k_stored"])
                                  / np.maximum(eatable, 1e-12), carry_onset)
    stock_base = eatable * me_mult[:, 1]                                             # (C,K) eatable onset stock per unit of extra share
    if theta.get("no_onset_stock"):
        stock_base = np.zeros_like(stock_base)
    stock0 = stock_base * extra_crop[:, 0]                                           # (C,K) at the pre-catastrophe herd

    p_cap = None
    if theta.get("no_production_when_freezing"):
        fm = z["freezing_months"][list(z["scenarios"]).index(tg)]                     # (C, group, T+1)
        if not shock:
            fm = hold(fm, 2)
        p_cap = 1.0 - (1 - theta["shelter"]) * np.where(LARGE[None, :, None], fm[:, :1, :], fm[:, 1:, :]) / 12.0
    # co-products of crushing, distilling and sugar making follow the country's crop calories
    tcal = I.crop_tcal(tg).sum(axis=1)                                               # (C, T+1)
    coproduct_ratio = np.where(tcal[:, :1] > 0, tcal[:, 1:] / np.maximum(tcal[:, :1], 1e-9), 1.0)
    if not shock:
        coproduct_ratio = np.ones((C, T))
    carcass = h["lw_kg"] * h["dressing"] * np.nan_to_num(h["carcass_kcal_per_kg"], nan=2000.0)
    growth_region = np.clip(h["growth_max"] * theta["growth_mult"], 0.0, h["growth_high"])
    extras = dict(onset_month=onset_month, uptake=uptake, access=access, access_crop=crop_access, eat_mu0=mu0, reserve=reserve, extra_share=extra_share, extra_share_crop=extra_crop, e_res=e_res, res_dm=dm.sum(axis=1),
                  me_head=me_head, dm_head=dm_head, cp_short=cp_short, raw_mp_m=raw_mp_m, raw_mp_p=raw_mp_p, raw_ndf_cap=cap, live_weight=h["lw_kg"], mp_head0=mp_head0, tdn_discount=discount, treated_fraction=by_crop(treated.astype(float)), residue_me_multiplier=me_mult, ndf_floor=ndf_floor, mp_alpha=mp_alpha, ndf_head0=ndf_head0, q0=q0,
                  residue_cap=cap_f, grain_cap=grain_f, tau0=tau0, tops_kept=tops_f, ration_uptake=np.where(produced0 > 0, claimed / np.maximum(produced0, 1e-9), np.inf),
                  base_crop_res=base_crop_res, growth=growth_region, inyear=inyear, untreated=untreated)
    return Prepared(
        iso3=iso, herds=I.HERDS, n0=n0, ne_m=ne_m, ne_l=ne_l, ne_g=ne_g, ne_p=ne_p, mp_m=mp_m, mp_p=mp_p, ndf_cap=ndf_cap,
        me=me, cp=cp, ndf=ndf, cold=cold, e0=e0, shares=shares, conc_share=split,
        grass_ratio=I.grass_ratio(tg) if shock else np.ones((C, T)), res_ratio=res_ratio,
        res_share=res_share, pool=pool, stock0=stock0, carry=carry[:, 1:, :], inyear=inyear[:, 1:, :], carry_onset=carry_onset, milk0=s["milk0"], meat0=s["meat0"],
        carcass_kcal=carcass, growth=growth_region, coproduct_ratio=coproduct_ratio, grain_kcal_per_mj=(st["grain_kcal_per_mj"] + st["bran_kcal_per_mj"]) * st["concentrates_me"] / me[:, :, 0, CONC],   # grain means grains and bran; food per kg is fixed, so per MJ it follows the ME drawn
        protein_model=model, rdp=rdp, drup=drup, tdn=tdn, treatment_npn=npn, urea_max_dm=urea_max_dm, p_cap=p_cap, extras=extras,
        scenario=tg, theta=theta, res_order=res_order, res_comp=comp, res_use=res_use,
        access_curve=curve, access_ratios=ratios, tau=tau, uptake=uptake, crop_uptake=crop_uptake, access_mult=access_mult,
        pool_usual=pool_usual, extra_base=extra_base, stock_base=stock_base,
        carry_stored=carried_parts["carry_stored"][:, 1:], carry_onset_stored=carry_onset_stored,
        inyear_carried=carried_parts["inyear_carried"][:, 1:], keep_carried=carried_parts["keep_carried"][:, 1:],
        species_weight=species_weight, herd_species=herd_species, eat_share=eat,
        reach_scale=scale, crop_curve=crops_g, crop_scale=crop_scale,
    )


def no_shock(prep: Prepared) -> Prepared:
    """The same country set with every driver held at its pre-catastrophe value:
    the validation case, and the potential of the redirection in normal times."""
    if prep.scenario is None:
        raise ValueError("no_shock requires a Prepared object returned by prepare()")
    # Rebuild with pre-catastrophe drivers before applying intervention levers. Holding
    # year-zero feed values afterward would silently switch straw treatment off.
    return prepare(prep.scenario, prep.theta, shock=False)
