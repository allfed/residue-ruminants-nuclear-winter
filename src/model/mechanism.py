"""The mechanism: rations under three constraints, herd choice, carry.

All arrays are vectorised over countries (C), herds (H) and, inside the
yearly solve, candidate herd sizes (J). Feeds (F) are ordered
grass, one field residue per counted crop (RES_CROPS, the crops of the crop
tables with a residue fed; each keeps its own feed values and treatment),
concentrates (grains and bran), and the other co-products of
crop processing (oilseed meals, distillers grains, molasses, pulp), which
stay in the ration in both cases and follow their crops. Residues are
taken in decreasing ME per kg NDF (ties: higher ME), an order set per
country and year from the feed values offered (res_order). A herd with
more feed than it needs eats the co-products first, then grass, then grain
and bran, and the residues last. Energy is
metabolisable energy (ME, MJ); animal
needs are net energy (NE) turned into ME with the IPCC 2019 efficiencies
(REM, REG) of the diet actually eaten, so a straw diet is used less efficiently than
the pre-catastrophe diet.

The additional residue offered each year is the reach allocation (residue_access.csv)
at the surviving herd: the country's demand multiplier mu (surviving heads
times the residue ME each can eat, its NDF-limited residue-only intake but
no more than its pre-catastrophe requirement, summed over herds, over the
national requirement of the whole herd) sets the share tau g(r / tau) of
the year's residue in reach, at the ratio r that puts what the herd can eat
and the national residue on the grid's patterns (access_share). The rations'
existing use is subtracted crop by crop within the country's budget (the
reach less the rations' share of the year's residue; crop_extra_share), and
no crop offers more than its own reach, the flow from that crop alone
(crop_access_share). Fixed herds know mu from the reference run;
free herds iterate reach and herd choice from the largest herd the year
allows until mu settles (access_tol), and fall back to the smallest mu seen
if it does not (access_rounds), which offers no more than the survivors
can reach.

Free herds choose surviving heads and ration for the most production under
energy, effective NDF intake and ration-level MP constraints, keeping the
largest herd within 0.5% of it (a sensitivity: the most milk and meat eaten,
each head kept beyond last year's herd costing its carcass). Kept-concentrate interventions match
reference heads, eat at most the usual feeds the reference herd ate plus
the extra residue, and minimize grain subject to a floor on their production of milk
and meat (none may be lost by default). Every ration is an exact linear
program at the diet's ME content, searched over that content (ration.py).
Urea is an explicit, capped ration ingredient; protein.py gives the MP rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.inputs import counted_crops
from src.model import params as P, protein, ration
from typing import Dict, Optional

import numpy as np

RES_CROPS = counted_crops()                 # crop codes of the residue feeds
K_RES = len(RES_CROPS)
GRASS = 0
RES = tuple(range(1, K_RES + 1))
RES_SLICE = slice(1, K_RES + 1)
CONC, COP = K_RES + 1, K_RES + 2
N_FEEDS = K_RES + 3
GE_FEED = 18.45         # MJ gross energy per kg DM (IPCC 2019 Refinement, Vol. 4 Ch. 10, p. 10.30)
DAYS = 365.0
BINDING = ("none", "supply", "intake", "protein", "herd gone", "regrowth limit")
STRATEGIES = ("output", "survival", "eaten")    # free herds: most production, most heads, most milk and meat eaten


@dataclass
class Prepared:
    """Everything the year loop needs, for one scenario and parameter draw."""

    iso3: list
    herds: tuple
    n0: np.ndarray            # (C,H) pre-catastrophe heads
    ne_m: np.ndarray          # (C,H) MJ NE/d per head, maintenance incl. activity
    ne_l: np.ndarray          # (C,H) lactation at pre-catastrophe yield
    ne_g: np.ndarray          # (C,H) growth
    ne_p: np.ndarray          # (C,H) pregnancy
    mp_m: np.ndarray          # (C,H) kg MP/d per head at maintenance
    mp_p: np.ndarray          # (C,H) kg MP/d per head for pre-catastrophe production
    ndf_cap: np.ndarray       # (C,H) kg NDF/d per head
    me: np.ndarray            # (C,H,T+1,F) MJ/kg DM
    cp: np.ndarray            # (C,H,T+1,F) kg/kg DM
    ndf: np.ndarray           # (C,H,T+1,F) kg/kg DM
    cold: np.ndarray          # (C,H,T+1) maintenance multiplier, year 0 = pre-catastrophe
    e0: np.ndarray            # (C,H) pre-catastrophe ME intake, MJ/yr
    shares: np.ndarray        # (C,H,3) pre-catastrophe energy shares grass, residues, concentrates (grains, bran and co-products together)
    conc_share: np.ndarray    # (C,H) share of the concentrate energy that is grains and bran; the rest is co-products
    grass_ratio: np.ndarray   # (C,T)
    res_ratio: np.ndarray     # (C,T) residue ME produced over pre-catastrophe
    res_share: np.ndarray     # (C,T+1,K) split of residue ME produced by crop
    pool: np.ndarray          # (C,T,K) residue ME the intervention can feed, MJ/yr
    stock0: np.ndarray        # (C,K) extra residue ME standing at onset that can be eaten in the twelve months from onset
    carry: np.ndarray         # (C,T,K) residue left after its twelve months of feeding per unit of eatable residue left uneaten
    milk0: np.ndarray         # (C,H) pre-catastrophe dietary kcal/yr
    meat0: np.ndarray         # (C,H)
    carcass_kcal: np.ndarray  # (C,H) dietary kcal per culled head
    growth: np.ndarray        # (C,H) maximum net herd growth per year
    coproduct_ratio: np.ndarray        # (C,T) co-product supply over pre-catastrophe: the country's crop calories
    grain_kcal_per_mj: np.ndarray      # (C,H) food kcal of the grain and the bran per MJ ME of concentrates (grains and bran)
    protein_model: str = "mp"
    rdp: Optional[np.ndarray] = None
    drup: Optional[np.ndarray] = None
    tdn: Optional[np.ndarray] = None
    treatment_npn: Optional[np.ndarray] = None
    urea_max_dm: float = 0.0
    p_cap: Optional[np.ndarray] = None   # (C,H,T+1) ceiling on the production level (structural switch)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)
    inyear: Optional[np.ndarray] = None         # (C,T,K) share of the residue entering a year's feeding that can be eaten (decay while fed)
    carry_onset: Optional[np.ndarray] = None    # (C,K) carry for the onset stock, whose twelve months start at onset
    carry_stored: Optional[np.ndarray] = None   # (C,T,K) the baled or stacked part of carry; the rest is field residue
    carry_onset_stored: Optional[np.ndarray] = None   # (C,K) the same for carry_onset
    inyear_carried: Optional[np.ndarray] = None  # (C,T,K,2) share of the residue carried into a year, baled or stacked and field, that can be eaten
    keep_carried: Optional[np.ndarray] = None    # (C,T,K,2) share of it left at the end of the year; None: carried residue fed as the year's harvest
    scenario: Optional[int] = None
    theta: dict = field(default_factory=dict)
    res_order: Optional[np.ndarray] = None   # (C,T+1,K) residue fill order, crop positions by rank
    access_curve: Optional[np.ndarray] = None   # (T+1,C,ratio) g(r) of the reach table at the draw's radius; None: all removable residue in reach
    access_ratios: Optional[np.ndarray] = None  # (ratio,) the tabulated demand ratios
    tau: Optional[np.ndarray] = None            # (C,) share of the residue that may leave the field
    uptake: Optional[np.ndarray] = None         # (C,) share of the residue the rations already contain (inf: nothing produced)
    crop_uptake: Optional[np.ndarray] = None    # (C,K) share of each crop's residue the rations already contain, every year (inf: nothing produced)
    access_mult: float = 1.0                    # collection-shortfall multiplier on the allocated share
    pool_usual: Optional[np.ndarray] = None     # (C,T,K) the rations' residue ME, MJ/yr
    extra_base: Optional[np.ndarray] = None     # (C,T,K) extra residue ME per unit of extra share (production, decay, treatment)
    stock_base: Optional[np.ndarray] = None     # (C,K) onset stock ME per unit of extra share
    species_weight: Optional[np.ndarray] = None # (C,3) national pre-catastrophe requirement of cattle, buffaloes, small ruminants (MJ/yr, the draw's)
    reach_scale: Optional[np.ndarray] = None    # (C,T+1) table ratio per unit of mu, E0 E_grid / (S D_grid); None: mu itself
    crop_curve: Optional[np.ndarray] = None     # (C,K,ratio) g_k(r) of each crop alone, year-0 grid, at the draw's radius; None: the country's reach for every crop
    crop_scale: Optional[np.ndarray] = None     # (C,T+1,K) crop k's table ratio per unit of mu, E0 E_grid_k(year 0) / (S_k D_grid)
    herd_species: Optional[np.ndarray] = None   # (H,) species index of each herd
    eat_share: Optional[np.ndarray] = None      # (C,H,T+1) residue ME a head can eat (NDF-limited residue-only diet) over its need, at most 1
    res_comp: Optional[np.ndarray] = None       # (C,H,K) the pre-catastrophe ration's residue ME by crop, shares
    res_use: Optional[np.ndarray] = None        # (C,H,T,K) the rations' residue ME per herd and crop, untreated, MJ/yr at the pre-catastrophe herd
    # sensitivity, residue shared between countries (runs/pooling.py); None: each country offers its own residue
    residue_floor: Optional[np.ndarray] = None  # (C,T,K) least additional residue ME offered, MJ/yr (the bound where residue never limits)
    residue_out: Optional[np.ndarray] = None    # (C,T,K) the country's own residue ME that leaves for the world pool, MJ/yr
    residue_in: Optional[np.ndarray] = None     # (C,T,K) its share of the world pool, MJ/yr


def residue_order(me, ndf):
    """(…,K) residue positions by rank: decreasing ME per kg NDF, ties by higher ME."""
    return np.lexsort((-me, -me / np.maximum(ndf, 1e-9)), axis=-1)


efficiencies = ration.efficiencies     # IPCC 2019 efficiencies of ME use (REM, REG) from the diet's M/D


def animal_output(prep, n, p, n_prev):
    """Reported milk and meat, including meat retained for rebuilding."""
    rel = n * p / np.maximum(prep.n0, 1e-9)
    retained = np.maximum(n - n_prev, 0.0) * prep.carcass_kcal
    return prep.milk0 * rel + np.maximum(prep.meat0 * rel - retained, 0.0)


def production_floor(prep, n, n_prev, target):
    """Invert the piecewise-linear reported animal-food function at fixed heads."""
    rel = n / np.maximum(prep.n0, 1e-9)
    milk, meat = prep.milk0 * rel, prep.meat0 * rel
    retained = np.maximum(n - n_prev, 0.0) * prep.carcass_kcal
    return np.where(target > 0, np.minimum(target / np.maximum(milk, 1e-30),
                    (target + retained) / np.maximum(milk + meat, 1e-30)), 0.0)


def demand_multiplier(prep: Prepared, n, t: int):
    """(C,) residue ME that heads n (C,H) can eat in year t (0 = pre-catastrophe), as a share of the
    country's pre-catastrophe requirement: each species is culled uniformly within the country, dairy and
    meat herds of a species share its map, each head eats at most its NDF-limited residue-only intake
    (prep.eat_share), and the species are weighted by prep.species_weight, their national pre-catastrophe
    requirement, so that mu times the national requirement is what the heads can eat (Cap)."""
    C, H = prep.n0.shape
    if prep.herd_species is None or prep.species_weight is None:
        return np.ones(C)
    per_head = np.where(prep.n0 > 0, prep.e0 / np.maximum(prep.n0, 1e-9), 0.0)          # pre-catastrophe MJ per head per year
    if prep.eat_share is not None:
        per_head = per_head * prep.eat_share[:, :, t]
    w, w0 = np.asarray(n, float) * per_head, prep.e0
    mu = np.zeros(C)
    total = np.zeros(C)
    for k in range(prep.species_weight.shape[1]):
        cols = prep.herd_species == k
        have = w0[:, cols].sum(axis=1)
        m_k = np.where(have > 0, w[:, cols].sum(axis=1) / np.maximum(have, 1e-9), 1.0)   # a species the model has no herd for is not culled
        weight = prep.species_weight[:, k]
        mu += weight * m_k
        total += weight
    return np.where(total > 0, mu / np.maximum(total, 1e-30), 1.0)


def access_share(prep: Prepared, t: int, mu):
    """(C,) share of year t's residue (0 = the pre-catastrophe harvest) in reach of a herd at demand
    multiplier mu: tau g(mu s / tau), linear between the tabulated ratios, times the shortfall multiplier.
    s (prep.reach_scale) puts the national residue S and what the herd can eat, Cap = mu E0, on the grid's
    patterns: the flow is homogeneous in its capacities, so S tau g(r / tau) at r = (Cap / S) (E_grid / D_grid)
    is the flow of those national levels and never exceeds Cap."""
    C = prep.n0.shape[0]
    tau = np.ones(C) if prep.tau is None else np.clip(prep.tau, 0.0, 1.0)
    mu = np.clip(np.asarray(mu, float), 0.0, None)
    if prep.access_curve is None:
        share = tau * (mu > 0)
    else:
        if prep.reach_scale is not None:
            mu = mu * prep.reach_scale[:, t]
        r = np.divide(mu, tau, out=np.zeros(C), where=tau > 0)
        grid = np.concatenate([[0.0], prep.access_ratios])
        rows = np.concatenate([np.zeros((C, 1)), prep.access_curve[t]], axis=1)
        share = tau * np.clip([np.interp(rc, grid, row) for rc, row in zip(r, rows)], 0.0, 1.0)   # integer flow units can round past one
    return share * prep.access_mult


def crop_reach_share(curves, ratios, tau, x):
    """(C,K) tau g_k(x / tau) from each crop's curve (C,K,ratio) on the ratio grid, tau (C,) and the table
    ratios x (C,K), interpolated as access_share: linear in the ratio, proportional below the first ratio,
    constant above the last, g at most one."""
    C, K = x.shape
    tau = np.repeat(np.clip(np.asarray(tau, float), 0.0, 1.0), K)
    r = np.divide(np.clip(x, 0.0, None).ravel(), tau, out=np.zeros(C * K), where=tau > 0)
    grid = np.concatenate([[0.0], ratios])
    rows = np.concatenate([np.zeros((C * K, 1)), curves.reshape(C * K, -1)], axis=1)
    return (tau * np.clip([np.interp(rc, grid, row) for rc, row in zip(r, rows)], 0.0, 1.0)).reshape(C, K)


def crop_access_share(prep: Prepared, t: int, mu):
    """(C,K) share of each crop's year-t residue in reach of a herd at demand multiplier mu, the crop alone
    (residue_access_crops.npz, year-0 grid): tau g_k(mu s_k / tau) times the shortfall multiplier, s_k = prep.crop_scale.
    Without the per-crop curves, every crop takes the country's share (access_share)."""
    if prep.crop_curve is None or prep.access_curve is None:
        return np.repeat(access_share(prep, t, mu)[:, None], K_RES, axis=1)
    C = prep.n0.shape[0]
    tau = np.ones(C) if prep.tau is None else prep.tau
    x = np.clip(np.asarray(mu, float), 0.0, None)[:, None] * prep.crop_scale[:, t]
    return crop_reach_share(prep.crop_curve, prep.access_ratios, tau, x) * prep.access_mult


def crop_extra_share(reach, uptake, weights, crop_reach=None):
    """(C,K) share of each crop's production offered as additional feed, from the country's share in reach
    (C,), each crop's own share in reach (crop_reach, C,K; default the country's), the share of each crop the
    rations already contain (C,K; inf: nothing produced) and the year's production split by crop (C,K, summing
    to one). The country's budget is its reach less the rations' share of the whole production, never
    negative (the existing use is kept even where it exceeds the reach); it goes to the crops in proportion to
    each crop's production times its room, its own reach less its share in the rations. A crop out of reach
    offers nothing; where the rooms add up to less than the budget, less is offered. A crop's usual and
    additional residue together exceed its own reach only where its usual residue alone does."""
    reach = np.asarray(reach, float)
    w = np.asarray(weights, float)
    u = np.broadcast_to(np.asarray(uptake, float), w.shape)
    used = np.multiply(u, w, out=np.zeros(w.shape), where=w > 0).sum(axis=1)       # the rations' share of the year's residue
    budget = np.maximum(reach - used, 0.0)
    room = np.maximum((reach[:, None] if crop_reach is None else np.asarray(crop_reach, float)) - u, 0.0)
    have = (room * w).sum(axis=1)                                                  # at least the budget with the country's reach
    return room * np.minimum(np.divide(budget, have, out=np.zeros_like(budget), where=have > 0), 1.0)[:, None]


def extra_share(prep: Prepared, t: int, mu):
    """(C,K) share of each crop's year-t residue production (0 = the pre-catastrophe harvest) offered as
    additional feed: what is in reach of the surviving herd minus what the rations already contain, crop by
    crop within the country's budget, no crop beyond its own reach (crop_extra_share). Without
    prep.crop_uptake, the national share (prep.uptake) is subtracted from every crop."""
    reach = access_share(prep, t, mu)
    if prep.crop_uptake is None:
        uptake = np.zeros(prep.n0.shape[0]) if prep.uptake is None else np.where(np.isfinite(prep.uptake), prep.uptake, 1e9)
        return np.repeat(np.maximum(reach - uptake, 0.0)[:, None], prep.res_share.shape[-1], axis=1)
    crop_reach = crop_access_share(prep, t, mu) if prep.crop_curve is not None else None
    return crop_extra_share(reach, prep.crop_uptake, prep.res_share[:, t], crop_reach)


def extra_flow(prep: Prepared, t: int, mu):
    """(C,K) additional residue ME offered in year t (1..T) at demand multiplier mu; prep.pool holds it at the pre-catastrophe herd."""
    if prep.extra_base is None:
        flow = np.maximum(prep.pool[:, t - 1, :] - prep.pool_usual[:, t - 1, :], 0.0) if prep.pool_usual is not None else prep.pool[:, t - 1, :] * 0
    else:
        flow = prep.extra_base[:, t - 1, :] * extra_share(prep, t, mu)
    if prep.residue_floor is not None:
        flow = np.maximum(flow, prep.residue_floor[:, t - 1, :])
    if prep.residue_in is not None:
        flow = np.maximum(flow - prep.residue_out[:, t - 1, :], 0.0) + prep.residue_in[:, t - 1, :]
    return flow


def onset_stock(prep: Prepared, mu):
    """(C,K) residue ME standing on the fields at the onset that the first year's herd can reach (the
    pre-catastrophe harvest, year-0 grid and crop split); prep.stock0 holds it at the pre-catastrophe herd."""
    if prep.stock_base is None:
        return prep.stock0
    return prep.stock_base * extra_share(prep, 0, mu)


def solve_year(prep: Prepared, t: int, n_avail, avail, *, extra=None, displace=False, dynamics=True,
               strategy="output", n_fixed=None, n_prev=None, reference_year=None, output_floor=0.0,
               tol=ration.Q_TOL, hint=None, scan=ration.STEP):
    """One year for all countries and herds: every ration, and every free herd's size, is optimal.

    ``avail`` (C,H,F) and ``extra`` (C,H,K) are annual ME allowances, residues
    at the year's (treated) values. Free herds (``n_fixed`` None) choose their
    heads up to ``n_avail`` and their ration for the most production, keeping
    the largest herd within 0.5% of it (``strategy="survival"``: the most heads
    fed; ``strategy="eaten"``: the most milk and meat eaten, each head kept
    beyond ``n_prev`` costing its carcass). With ``displace`` and grain in the ration, a free herd counts the
    grain it frees as food. Fixed herds keep ``n_fixed`` heads: with
    ``displace`` they eat the least grain that keeps their production of milk
    and meat at least (1 - ``output_floor``) times that of ``reference_year``
    (None: no floor) and never more, then the most milk and meat, from at most the usual
    feeds the reference herd ate plus the extra residue; without
    ``displace`` they keep the reference grain and maximise output.
    ``dynamics=False`` is the proportional rule: all available heads, output
    proportional to feed energy, protein ignored. Where treated straw is
    offered, its untreated form is too, and the least urea is used. Return
    heads, production, annual ME and DM eaten by feed (residue: usual and
    extra together), treated straw DM, the extra and usual residue eaten
    (ME at the offered values), supplemental urea (kg/yr), the diet's M/D,
    the binding limit and feasibility (``nutrients_ok``: the ration returned meets energy, MP
    and NDF, checked apart from the solver; fixed herds need it). ``hint`` (C,H): each ration's M/D in an earlier,
    similar solve, from which the search over it starts; ``tol`` and ``scan``: the search's
    final width and coarse spacing (MJ per kg DM).
    """
    C, H = n_avail.shape
    fixed = n_fixed is not None
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be output, survival or eaten")
    if output_floor is not None and not 0 <= output_floor <= 1:
        raise ValueError("Allowed output loss must be between zero and one, or None")
    usual_mode = prep.theta.get("usual_residue", "ranked")
    if usual_mode not in ("mix", "ranked"):
        raise ValueError("usual_residue must be mix or ranked")
    n_ref = np.asarray(n_fixed, float) if fixed else (n_avail if dynamics else n_avail)
    n_prev = n_avail if n_prev is None else n_prev
    live = n_ref > 0
    out = dict(n=np.zeros((C, H)), p=np.zeros((C, H)), q=np.full((C, H), np.nan), used=np.zeros((C, H, N_FEEDS)), dm=np.zeros((C, H, N_FEEDS)),
               treated_dm=np.zeros((C, H, K_RES)), extra_used=np.zeros((C, H, K_RES)), usual_res_used=np.zeros((C, H, K_RES)),
               urea=np.zeros((C, H)), md=np.zeros((C, H)), binding=np.full((C, H), BINDING.index("herd gone")),
               fixed_feasible=np.ones((C, H), bool), diet_settled=np.ones((C, H), bool), nutrients_ok=np.ones((C, H), bool))
    if not live.any():
        return out
    c_, h_ = np.nonzero(live)
    N = n_ref[c_, h_]
    per = N * DAYS                                                     # reference head-days
    me, ndf, cp = prep.me[c_, h_, t], prep.ndf[c_, h_, t], prep.cp[c_, h_, t]
    mp, mp_urea = protein.rows(prep.protein_model, prep.tdn[c_, h_, t], prep.rdp[c_, h_, t], prep.drup[c_, h_, t], cp)
    nb = protein.added_n_room(prep, t)[c_, h_]                         # room for supplemental urea N (retained treatment N does not count)
    usual = avail[c_, h_] / me / per[:, None]                          # kg DM per reference head per day
    usual_res = usual[:, RES_SLICE].copy()
    if fixed and reference_year is not None and "dm" in reference_year:
        cap_ref = reference_year["dm"][c_, h_] / per[:, None]            # what the reference herd ate
        usual = np.minimum(usual, cap_ref * (1 + 1e-9) + 1e-12)
        usual_res = usual[:, RES_SLICE].copy()
    ext = extra[c_, h_] / me[:, RES_SLICE] / per[:, None] if extra is not None else np.zeros_like(usual_res)
    per_head0 = prep.e0 * prep.shares[:, :, 2] / np.maximum(prep.n0, 1e-9) / DAYS
    g0 = (per_head0 * prep.conc_share / prep.me[:, :, 0, CONC])[c_, h_]
    # feed columns: the prep layout, residues offered usual and extra together; then untreated straw
    ub = usual.copy()
    ub[:, RES_SLICE] = usual_res + ext
    lb = np.zeros_like(ub)
    order = prep.res_order[c_, t] if prep.res_order is not None else residue_order(me[:, RES_SLICE], ndf[:, RES_SLICE])
    rank = np.argsort(order, axis=1)
    tier = np.zeros_like(ub)
    # spare feed is left in this order: the co-products and grass are eaten before grain and bran, so
    # grain that surplus grass could replace is not freed only by residue in the kept case
    tier[:, COP], tier[:, GRASS], tier[:, CONC] = 1, 2, 3
    tier[:, RES_SLICE] = 4 + rank
    untreated = prep.extras.get("untreated") if prep.extras else None
    eligible = (prep.extras["treated_fraction"][c_, t] > 0) if untreated is not None else np.zeros_like(usual_res, bool)
    treated = np.zeros_like(ub)
    treated[:, RES_SLICE] = eligible
    pairs = []
    cols = [me, ndf, mp, nb, lb, ub, tier, treated]
    if untreated is not None:
        u = {k: v[c_, h_, t] for k, v in untreated.items()}
        mp_u, _ = protein.rows(prep.protein_model, u["tdn"], u["rdp"], u["drup"], u["cp"])
        ub_u = np.where(eligible, ub[:, RES_SLICE], 0.0)
        cols = [np.concatenate([me, u["me"]], 1), np.concatenate([ndf, u["ndf"]], 1), np.concatenate([mp, mp_u], 2),
                np.concatenate([nb, protein.MAX_ADDED_N_SHARE * u["cp"]], 1), np.concatenate([lb, np.zeros_like(ub_u)], 1),
                np.concatenate([ub, ub_u], 1), np.concatenate([tier, tier[:, RES_SLICE]], 1), np.concatenate([treated, np.zeros_like(ub_u)], 1)]
        pairs = [(RES[k], N_FEEDS + k) for k in range(K_RES)]
    me_c, ndf_c, mp_c, nb_c, lb_c, ub_c, tier_c, treated_c = cols
    if usual_mode == "mix" and untreated is None:
        # the usual residue as one mixture in the proportions offered, the extra residue by crop
        mix = usual_res                                                 # kg DM per reference head at the full mixture
        tot = mix.sum(1, keepdims=True)
        w = np.where(tot > 0, mix / np.maximum(tot, 1e-30), 0.0)
        mcol = lambda a: (a[:, RES_SLICE] * w).sum(1, keepdims=True) if a.ndim == 2 else (a[:, :, RES_SLICE] * w[:, None]).sum(2, keepdims=True)
        me_c, ndf_c, nb_c = (np.concatenate([a, mcol(a)], 1) for a in (me_c, ndf_c, nb_c))
        mp_c = np.concatenate([mp_c, mcol(mp_c)], 2)
        ub_c = ub_c.copy(); ub_c[:, RES_SLICE] = ext
        ub_c = np.concatenate([ub_c, tot], 1)
        lb_c = np.concatenate([lb_c, np.zeros_like(tot)], 1)
        tier_c = np.concatenate([tier_c, np.full_like(tot, 4.0)], 1)
        treated_c = np.concatenate([treated_c, np.zeros_like(tot)], 1)
    elif usual_mode == "mix":
        raise ValueError("the fixed-mixture switch is not combined with treated straw")
    if fixed and not displace and reference_year is not None and "dm" in reference_year:
        lb_c[:, CONC] = ub_c[:, CONC]                                   # the grain stays: residues can only add output
    floor = np.zeros(len(N))
    if fixed and displace and reference_year is not None and output_floor is not None:
        # the same heads produce at least (1 - loss) of their milk and meat without the intervention. Meat kept to
        # rebuild is a fixed deduction, so the production level is found by inverting the reported output; where none
        # is reported (all of it rebuilds the herd), production itself loses at most the same share.
        target = (1 - output_floor) * animal_output(prep, n_fixed, reference_year["p"], n_prev)
        floor = np.maximum(production_floor(prep, n_fixed, n_prev, target)[c_, h_],
                           (1 - output_floor) * reference_year["p"][c_, h_])
    if fixed:
        objective = "grain" if displace else "output"
    elif not dynamics:
        objective = "output"
    elif strategy == "survival":
        objective = "survival"
    else:
        objective = "food" if (displace and extra is not None and np.any(ub_c[:, CONC] > 0)) else "output"
    pcap = prep.p_cap[c_, h_, t] if prep.p_cap is not None else np.ones(len(N))
    if fixed and displace and reference_year is not None and "p" in reference_year:
        pcap = np.minimum(pcap, reference_year["p"][c_, h_])            # the residue replaces grain, it does not add output
    outk = ((prep.milk0 + prep.meat0) / np.maximum(prep.n0, 1e-9) / DAYS)[c_, h_]
    gk = prep.grain_kcal_per_mj[c_, h_] * me[:, CONC]
    eaten = strategy == "eaten" and dynamics and not fixed               # heads beyond last year's cost their carcasses
    bt = ration.Batch(me=me_c, ndf=ndf_c, mp=mp_c, mp_urea=mp_urea, nb=nb_c, lb=lb_c, ub=ub_c, tier=tier_c,
                      treated=treated_c, pairs=pairs, grain=CONC, g0=g0,
                      nem=prep.ne_m[c_, h_] * prep.cold[c_, h_, t], nel=prep.ne_l[c_, h_], neg=prep.ne_g[c_, h_],
                      nep=prep.ne_p[c_, h_], mpm=prep.mp_m[c_, h_], mpp=prep.mp_p[c_, h_], cap=prep.ndf_cap[c_, h_],
                      lw=prep.extras["live_weight"][c_, h_] if prep.extras and "live_weight" in prep.extras else np.zeros(len(N)),
                      pcap=np.minimum(pcap, 1.0), free=dynamics and not fixed, floor=floor, objective=objective,
                      outk=outk, gk=gk, proportional=not dynamics, urea_max_dm=prep.urea_max_dm,
                      prev=np.asarray(n_prev, float)[c_, h_] / N if eaten else None,
                      carcass=prep.carcass_kcal[c_, h_] / DAYS if eaten else None,
                      system=prep.theta.get("efficiency_system", "ipcc"))
    anchor = seed = None
    if fixed and reference_year is not None and "md" in reference_year:
        # the ration eaten without the intervention is feasible: the search starts from it (if it checks so)
        anchor = reference_year["md"][c_, h_]
        x0 = np.zeros((len(N), ub_c.shape[1] + (1 if prep.urea_max_dm > 0 else 0) + 2))
        ref_x = reference_year["dm"][c_, h_] / per[:, None]
        x0[:, :N_FEEDS] = ref_x
        if untreated is not None:                                        # the reference straw was untreated
            x0[:, RES_SLICE] = np.where(eligible, 0.0, ref_x[:, RES_SLICE])
            x0[:, N_FEEDS:N_FEEDS + K_RES] = np.where(eligible, ref_x[:, RES_SLICE], 0.0)
        if usual_mode == "mix" and untreated is None:                   # the reference ate its residue as the mixture, kg DM
            x0[:, RES_SLICE] = 0.0
            x0[:, ub_c.shape[1] - 1] = ref_x[:, RES_SLICE].sum(1)
        x0[:, :ub_c.shape[1]] = np.minimum(x0[:, :ub_c.shape[1]], ub_c)
        nu_col = ub_c.shape[1] + (1 if prep.urea_max_dm > 0 else 0)
        if prep.urea_max_dm > 0 and "urea" in reference_year:           # and its supplemental urea
            x0[:, ub_c.shape[1]] = reference_year["urea"][c_, h_] / per
        x0[:, nu_col] = 1.0
        ref_p = reference_year["p"][c_, h_] if "p" in reference_year else floor
        x0[:, nu_col + 1] = np.clip(np.maximum(ref_p, floor), 0.0, pcap)
        seed = (x0, anchor)
    r = bt.solve(tol=tol, within=None if (strategy == "survival" or not dynamics or fixed) else 0.005, anchor=anchor, seed=seed,
                 hint=None if hint is None else hint[c_, h_], step=scan)
    feeds = r["feeds"]
    dm = feeds[:, :N_FEEDS].copy()
    used = feeds[:, :N_FEEDS] * me
    tdm = np.where(eligible, feeds[:, RES_SLICE], 0.0)
    if untreated is not None:
        dm[:, RES_SLICE] += feeds[:, N_FEEDS:N_FEEDS + K_RES]
        used[:, RES_SLICE] += feeds[:, N_FEEDS:N_FEEDS + K_RES] * u["me"]
    if usual_mode == "mix" and untreated is None:
        mixed = feeds[:, -1:] * w                                        # the mixture's crops
        dm[:, RES_SLICE] += mixed
        used[:, RES_SLICE] += mixed * me[:, RES_SLICE]
        usual_eaten = mixed
    else:
        usual_eaten = np.minimum(dm[:, RES_SLICE], usual_res)            # the usual residue is eaten first
    extra_eaten = np.clip(dm[:, RES_SLICE] - usual_eaten, 0.0, ext)    # never more than the herd's allotment
    # solutions never reach zero exactly: a herd below one animal is gone
    kept = (r["nu"] * N >= 1.0) & (r["nu"] >= 1e-8)
    zero = lambda a: np.where(kept.reshape((-1,) + (1,) * (a.ndim - 1)), a, 0.0)
    dm, used, tdm, usual_eaten, extra_eaten = map(zero, (dm, used, tdm, usual_eaten, extra_eaten))
    r["urea"], r["p"], r["nu"] = zero(r["urea"]), zero(r["p"]), zero(r["nu"])
    out["n"][c_, h_] = r["nu"] * N
    out["p"][c_, h_] = r["p"]
    out["dm"][c_, h_] = dm * per[:, None]
    out["used"][c_, h_] = used * per[:, None]
    out["treated_dm"][c_, h_] = tdm * per[:, None]
    out["extra_used"][c_, h_] = extra_eaten * me[:, RES_SLICE] * per[:, None]
    out["usual_res_used"][c_, h_] = usual_eaten * me[:, RES_SLICE] * per[:, None]
    out["urea"][c_, h_] = r["urea"] * per
    out["md"][c_, h_] = r["md"]
    out["q"][c_, h_] = r["q"]
    p_sel, keep = r["p"], kept
    room = r["room"] <= 1e-6 * np.maximum(bt.cap * np.maximum(r["nu"], 1e-30), 1e-12)
    binding = np.where(p_sel >= 0.999, 0, np.where(r["p_mp"] < r["p_e"] - 1e-6, 3, np.where(room, 2, 1)))
    binding = np.where((r["nu"] < 0.999) & (binding == 0), 1, binding)
    if not fixed:                                                      # full output, but the herd is held below its
        held = N < prep.n0[c_, h_] * (1 - 1e-6)                          # pre-catastrophe size by the regrowth limit
        binding = np.where((binding == 0) & held, BINDING.index("regrowth limit"), binding)
    out["binding"][c_, h_] = np.where(keep, binding, BINDING.index("herd gone"))
    out["diet_settled"][c_, h_] = r["conv"]
    out["nutrients_ok"][c_, h_] = r["nutrients_ok"] | ~kept               # the ration returned meets energy, MP and NDF
    if fixed:
        out["fixed_feasible"][c_, h_] = r["feasible"] & (r["p"] >= np.minimum(floor, 1.0) - 1e-6) & out["nutrients_ok"][c_, h_]
    return out


def simulate(prep: Prepared, *, intervention: bool, concentrate_ratio, displace_grain: bool = True,
             herd_dynamics: bool = True, allocation: str = "pro rata", strategy: str = "output",
             reference: Optional[dict] = None, keep_year: Optional[int] = None,
             keep_crops: bool = False, herd: str = "auto", output_floor: Optional[float] = 0.0,
             coproducts_gone: bool = False, access_rounds: int = 8, access_tol: float = 2e-3,
             tol: float = ration.Q_TOL, scan: float = ration.STEP, release_usual: bool = True,
             share_rounds: int = 20, share_tol: float = 1e-4):
    """Ten years. ``concentrate_ratio`` is the share of the pre-catastrophe grain and
    bran ration still fed: a scalar (the cases: 1 kept, 0 gone) or a (C,T)
    array; a (C,H,T) array can specify a separate supply for each herd. The
    other co-products of crop processing (oilseed meals, distillers grains,
    molasses, pulp) stay in both cases and follow the country's crop
    production; ``coproducts_gone`` removes them too (a sensitivity:
    the gone case with the co-products gone as well). By default, interventions with grain
    available pin heads to ``reference``, eat at most the usual feeds the
    reference herd ate, and keep all of its production of milk and meat
    (``output_floor`` is the share that may be lost; None: no floor).
    ``herd="free"`` lets them choose their size. Gone-grain runs choose
    their herd size. Scarce extra residue is shared between the herds of a
    country, crop by crop, in proportion to what each would eat of the crop
    if it had all of it. The sharing is repeated: each round offers every
    herd its allotment so far plus all that is left, and shares what the
    herds want beyond their allotments in the same way, so a herd cut back
    on one crop can turn to the crops the others leave. It stops when no
    country's herds take more than ``share_tol`` of its supply in a round, or
    after ``share_rounds`` rounds (``share_rounds`` and ``share_converged``
    report it). The residue fed per crop (``res_crop_used``,
    and ``fed_by_crop``, ``fed_extra_by_crop`` in ``fate``) is always
    returned; ``keep_crops`` is accepted for older callers. The additional
    residue of each year is the reach allocation at the surviving herd
    (``demand_multiplier``); free herds iterate herd and reach up to
    ``access_rounds`` times until every country's multiplier moves by
    less than ``access_tol``, else take the smallest multiplier seen.
    ``access_rounds``, ``access_converged`` and ``access_mu`` report it.
    ``diet_settled`` flags the rations whose search converged (``tol`` and ``scan``: the
    search's final width and coarse spacing, MJ per kg DM). With ``release_usual``, the
    intervention offers each herd the usual residue its reference herd ate, and the rest of
    the usual residue joins the additional residue of the year (``usual_released`` in
    ``fate``), as does any usual residue the intervention herd leaves uneaten. The baled or stacked
    residue carried out of a year is at most ``theta["store_cap_years"]`` years of the additional
    residue the country's herds ate that year, in DM; the excess (``store_lost``) is counted in
    ``decayed``. ``strategy``: how free
    herds choose their size (solve_year).
    """
    if access_rounds < 1 or access_tol <= 0:
        raise ValueError("access_rounds must be positive and access_tol a positive tolerance")
    if share_rounds < 1 or share_tol <= 0:
        raise ValueError("share_rounds must be positive and share_tol a positive tolerance")
    C, H = prep.n0.shape
    T = prep.grass_ratio.shape[1]
    kr = np.asarray(concentrate_ratio, dtype=float)
    if kr.ndim == 2:
        kr = kr[:, None, :]
    kr = np.broadcast_to(kr, (C, H, T))
    if not np.isfinite(kr).all() or (kr < 0).any():
        raise ValueError("Concentrate ratios must be finite and nonnegative")
    if herd not in ("auto", "free", "reference"):
        raise ValueError("herd must be auto, free, or reference")
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be output, survival or eaten")
    store_cap = float(prep.theta.get("store_cap_years", P.SCALARS["store_cap_years"][0]))
    if not store_cap >= 0:
        raise ValueError("store_cap_years must be nonnegative (inf: no cap)")
    # with grain in the ration, the intervention keeps the reference herd (all available heads under the
    # proportional rule) and eats the least grain that keeps its milk and meat
    fixed = herd == "reference" or (herd == "auto" and intervention and np.any(kr > 0))
    if (fixed or (intervention and release_usual)) and reference is None:
        reference = simulate(prep, intervention=False, concentrate_ratio=kr, herd="free",
                             coproducts_gone=coproducts_gone, tol=tol, scan=scan)
    res = {k: np.zeros((C, H, T)) for k in ("n", "p", "milk", "meat", "carcass", "conc_used", "cop_used", "res_used", "grass_used", "cull", "urea")}
    res["binding"] = np.zeros((C, H, T), dtype=int)
    res["diet_used"] = np.zeros((C, H, T, N_FEEDS))
    res["diet_dm"] = np.zeros((C, H, T, N_FEEDS))
    res["treated_dm"] = np.zeros((C, H, T, K_RES))
    res["diet_md"] = np.zeros((C, H, T))
    res["diet_settled"] = np.ones((C, H, T), dtype=bool)
    feed_attrs = ("me", "cp", "ndf", "rdp", "drup", "tdn", "treatment_npn")
    for attr in feed_attrs:
        res["feed_" + attr] = getattr(prep, attr).copy()
    res["fate"] = {k: np.zeros((C, T)) for k in (
        "pool", "stock_in", "extra_flow", "usual_released", "usual_offered", "unused_usual",
        "fed", "fed_extra", "carried", "decayed", "stock_quality_change",
        "extra_flow_dm", "usual_released_dm", "stock_in_dm", "fed_extra_dm", "carried_dm", "decayed_dm",
        "carried_baled_dm", "store_lost", "store_lost_dm")}
    res["fate"].update({k: np.zeros((C, T, K_RES)) for k in ("fed_by_crop", "fed_extra_by_crop")})
    res["res_crop_used"] = np.zeros((C, H, T, K_RES))
    res["access_rounds"] = np.zeros(T, dtype=int)
    res["share_rounds"] = np.zeros(T, dtype=int)
    res["share_converged"] = np.ones((C, T), dtype=bool)
    res["access_converged"] = np.ones((C, T), dtype=bool)
    res["access_mu"] = np.ones((C, T))
    n_prev = prep.n0.copy()
    # Carry physical DM per crop. Stocks take the crop's composition of the
    # year when offered; quality revaluation is reported separately from decay.
    # stock_dm is the residue left at the end of its twelve months of feeding, in
    # its baled or stacked and field parts; each keeps its form the next year
    # (inyear_carried, keep_carried). Without them, one part fed as that year's harvest.
    if prep.inyear_carried is not None:
        c_parts = np.stack([prep.carry_stored, prep.carry - prep.carry_stored], axis=-1)                 # (C,T,K,2)
        c_onset = np.stack([prep.carry_onset_stored, prep.carry_onset - prep.carry_onset_stored], axis=-1)
        iy_c, keep_c = prep.inyear_carried, prep.keep_carried
    else:
        iy = prep.inyear if prep.inyear is not None else np.ones_like(prep.carry)
        c_parts, iy_c, keep_c = prep.carry[..., None], iy[..., None], (prep.carry * iy)[..., None]
        c_onset = (prep.carry_onset if prep.carry_onset is not None else prep.carry[:, 0])[..., None]
    stock_dm = np.zeros((C, K_RES, iy_c.shape[-1]))   # the onset stock enters in year 1 at that year's herd
    hint = np.full((C, H), np.nan)                     # each ration's M/D in the last solve: where the next search starts

    def solved(sol):
        hint[:] = np.where(np.isfinite(sol["q"]), sol["q"], hint)
        return sol

    for t in range(1, T + 1):
        crop_me = np.maximum(prep.me[:, 0, t, RES_SLICE], 1e-9)
        n_avail = np.minimum(prep.n0, n_prev * (1.0 + prep.growth)) if herd_dynamics else prep.n0
        avail = np.zeros((C, H, N_FEEDS))
        avail[..., GRASS] = prep.e0 * prep.shares[:, :, 0] * prep.grass_ratio[:, t - 1, None]
        conc_e0 = prep.e0 * prep.shares[:, :, 2]
        avail[..., CONC] = conc_e0 * prep.conc_share * kr[:, :, t - 1]
        avail[..., COP] = 0.0 if coproducts_gone else conc_e0 * (1 - prep.conc_share) * prep.coproduct_ratio[:, t - 1, None]
        if prep.res_use is not None:                                                         # each crop of the ration follows its own production
            avail[..., RES_SLICE] = prep.res_use[:, :, t - 1, :] * prep.extras["residue_me_multiplier"][:, t][:, None, :]
        else:
            base_res = prep.e0 * prep.shares[:, :, 1] * prep.res_ratio[:, t - 1, None]           # (C,H)
            avail[..., RES_SLICE] = base_res[..., None] * (prep.res_share[:, t] * prep.extras["residue_me_multiplier"][:, t])[:, None, :]
        n_ref = reference["n"][:, :, t - 1] if reference is not None else None
        ref_year = None
        if fixed:
            ref_year = dict(animal=(reference["milk"] + reference["meat"])[:, :, t - 1],
                            dm=reference["diet_dm"][:, :, t - 1],              # what the herd ate without the intervention
                            md=reference["diet_md"][:, :, t - 1], p=reference["p"][:, :, t - 1],
                            urea=reference["urea"][:, :, t - 1])                # supplemental urea, kg/yr
        common = dict(n_fixed=n_ref if fixed else None, n_prev=n_prev, reference_year=ref_year,
                      output_floor=output_floor, tol=tol, hint=hint, scan=scan)
        if not intervention:
            sol = solved(solve_year(prep, t, n_avail, avail, dynamics=herd_dynamics, strategy=strategy, **common))
        else:
            released = np.zeros((C, K_RES))
            release_lost = np.zeros((C, K_RES))
            if release_usual:
                # each herd keeps the usual residue its reference herd ate; the rest is additional residue, which
                # decays while it is fed like the year's additional residue (the eatable share inyear)
                ref_me = reference["diet_dm"][:, :, t - 1, RES_SLICE] * prep.me[:, :, t, RES_SLICE]
                kept_usual = np.minimum(avail[..., RES_SLICE], ref_me)
                released = (avail[..., RES_SLICE] - kept_usual).sum(axis=1)                  # (C,K)
                avail[..., RES_SLICE] = kept_usual
                if prep.inyear is not None:
                    release_lost = released * (1.0 - prep.inyear[:, t - 1, :])
                    released = released - release_lost
            base_k = avail[..., RES_SLICE].sum(axis=1)                                     # (C,K)
            groups = [list(range(H))] if allocation == "pro rata" else \
                [[h for h in range(H) if prep.herds[h].startswith("dairy")],
                 [h for h in range(H) if not prep.herds[h].startswith("dairy")]]
            kw = dict(displace=displace_grain, dynamics=herd_dynamics, strategy=strategy, **common)

            share_diag = {}

            def feed_year(mu):
                """Offer the extra residue in reach of a herd at multiplier mu and solve the year."""
                stock_dm_t = onset_stock(prep, mu) / np.maximum(prep.me[:, 0, 1, RES_SLICE], 1e-9) if t == 1 else \
                    (stock_dm * iy_c[:, t - 1]).sum(axis=-1)                                           # eatable
                stock_t = stock_dm_t * crop_me
                extra_t = extra_flow(prep, t, mu) + released
                supply_t = extra_t + stock_t                                                 # extra residue, (C,K)
                remaining = supply_t.copy()
                limit = np.zeros((C, H, K_RES))
                small = share_tol * supply_t.sum(axis=1) + 1e-9                              # (C,), a round that adds less ends the sharing
                rounds, settled = 0, np.ones(C, bool)
                for grp in groups:
                    # each round offers every herd its allotment plus all that is left; what the herds
                    # want beyond their allotments is shared crop by crop in proportion to those wants
                    for r in range(share_rounds):
                        offer = limit + remaining[:, None, :]
                        trial = solved(solve_year(prep, t, n_avail, avail, extra=offer, **kw))
                        want = np.maximum(trial["extra_used"][:, grp, :] - limit[:, grp, :], 0.0)   # beyond the allotment
                        total_want = want.sum(axis=1)                                        # (C,K)
                        enough = total_want <= remaining + 1e-6
                        share = want / np.maximum(total_want, 1e-30)[:, None, :]             # crop by crop
                        add = np.where(enough[:, None, :], want, remaining[:, None, :] * share)
                        limit[:, grp, :] += add
                        remaining = np.maximum(remaining - add.sum(axis=1), 0.0)
                        done = add.sum(axis=(1, 2)) <= small
                        if done.all():
                            break
                    rounds = max(rounds, r + 1)
                    settled &= done
                share_diag["rounds"], share_diag["settled"] = rounds, settled
                return solved(solve_year(prep, t, n_avail, avail, extra=limit, **kw)), extra_t, supply_t, stock_t, stock_dm_t

            # The reach is that of the surviving herd. Fixed herds know it; free herds iterate from the
            # largest herd the year allows, whose reach is the largest, and settle when mu stops moving.
            # A country whose herd flips between two candidate sizes is frozen at the smaller reach,
            # which the larger herd can use too; Aitken's extrapolation shortens slow contractions.
            coupled = prep.access_curve is not None and herd_dynamics and not fixed
            mu = demand_multiplier(prep, n_ref if fixed else n_avail, t)
            history, frozen, converged, rounds = [], np.zeros(C, bool), np.ones(C, bool), 0
            while True:
                rounds += 1
                sol, extra_t, supply, stock, stock_dm_in = feed_year(mu)
                if not coupled:
                    break
                mu_new = demand_multiplier(prep, sol["n"], t)
                converged = np.where(frozen, mu_new >= mu - access_tol, np.abs(mu_new - mu) <= access_tol)
                history.append((mu, mu_new))
                if converged.all() or rounds >= access_rounds:
                    break
                nxt = np.where(frozen | converged, mu, mu_new)                         # a settled country keeps its mu
                if len(history) >= 2:
                    (m0, f0), (m1, f1) = history[-2], history[-1]
                    cycle = ~frozen & ~converged & (np.abs(f1 - m0) <= access_tol)
                    frozen |= cycle
                    nxt = np.where(cycle, np.minimum(m0, m1), nxt)
                    d0, d1 = f0 - m0, f1 - m1
                    slope = np.divide(d1, d0, out=np.zeros(C), where=np.abs(d0) > access_tol / 10)
                    steady = ~frozen & ~converged & (slope >= 0.0) & (slope < 0.95) & (np.abs(d1) > access_tol)
                    jump = np.divide(d1, 1.0 - slope, out=np.zeros(C), where=steady)
                    nxt = np.where(steady, np.clip(m1 + jump, 0.0, 1.0), nxt)
                mu = nxt
            if coupled and not converged.all():
                # no settled herd: offer the reach of the smallest herd seen, which any larger herd can reach too
                low = np.minimum(np.min([m for m, _ in history], axis=0), np.min([f for _, f in history], axis=0))
                mu = np.where(converged, mu, low)
                sol, extra_t, supply, stock, stock_dm_in = feed_year(mu)
                rounds += 1
                converged = demand_multiplier(prep, sol["n"], t) >= mu - access_tol
            res["access_rounds"][t - 1] = rounds
            res["share_rounds"][t - 1] = share_diag["rounds"]
            res["share_converged"][:, t - 1] = share_diag["settled"]
            res["access_converged"][:, t - 1] = converged
            res["access_mu"][:, t - 1] = mu
            new_t = extra_flow(prep, t, mu)                                                  # the year's residue beyond the usual rations (not extra_t - released, which rounds)
            pool_t = prep.pool_usual[:, t - 1, :] + new_t if prep.pool_usual is not None else prep.pool[:, t - 1, :]
            fed_extra = sol["extra_used"].sum(axis=1)                                        # (C,K), at the offered values
            left = np.maximum(supply - fed_extra, 0.0)
            actual_base_k = base_k
            late = np.zeros_like(base_k)
            if release_usual:
                # usual residue the intervention herd leaves uneaten joins what is left of the additional residue
                late = np.maximum(base_k - sol["usual_res_used"].sum(axis=1), 0.0)
                released = released + late
                actual_base_k = base_k - late
            res["fate"]["pool"][:, t - 1] = pool_t.sum(axis=1)
            res["fate"]["stock_in"][:, t - 1] = stock.sum(axis=1)
            res["fate"]["stock_in_dm"][:, t - 1] = stock_dm_in.sum(axis=1)
            if t > 1:
                res["fate"]["stock_quality_change"][:, t-1] = stock.sum(1) - res["fate"]["carried"][:, t-2]
            res["fate"]["extra_flow"][:, t - 1] = new_t.sum(axis=1)
            res["fate"]["extra_flow_dm"][:, t-1] = (new_t/crop_me).sum(1)
            res["fate"]["usual_released"][:, t - 1] = (released + release_lost).sum(axis=1)
            res["fate"]["usual_released_dm"][:, t-1] = ((released + release_lost)/crop_me).sum(1)
            res["fate"]["usual_offered"][:, t - 1] = actual_base_k.sum(axis=1)
            res["fate"]["fed_by_crop"][:, t - 1] = (sol["usual_res_used"] + sol["extra_used"]).sum(axis=1)
            res["fate"]["fed_extra_by_crop"][:, t - 1] = fed_extra
            res["fate"]["fed"][:, t - 1] = res["fate"]["fed_by_crop"][:, t - 1].sum(axis=1)
            res["fate"]["fed_extra"][:, t - 1] = fed_extra.sum(axis=1)
            # Without release_usual, usual feed that a smaller herd leaves uneaten
            # is not redistributed or carried; with it, it is part of the additional residue.
            usual_used = res["fate"]["fed"][:, t - 1] - fed_extra.sum(axis=1)
            res["fate"]["unused_usual"][:, t - 1] = actual_base_k.sum(axis=1) - usual_used
            if keep_year == t:
                res["kept"] = dict(avail=avail.copy(), supply=supply.copy(), n=sol["n"].copy(), p=sol["p"].copy(), md=sol["md"].copy())
            # Eating takes the same share of every part of the supply; what is left
            # uneaten decays to the end of its twelve months (carry, by part: the
            # stock carried in keeps its form), then is counted at what the next
            # year can eat of it (inyear_carried).
            frac = np.divide(left, supply, out=np.zeros_like(left), where=supply > 0)        # uneaten share of the supply
            stock_end = stock[..., None] * c_onset if t == 1 else stock_dm * keep_c[:, t - 1] * crop_me[..., None]
            end = (frac * extra_t + late)[..., None] * c_parts[:, t - 1] + frac[..., None] * stock_end   # (C,K,part)
            left = left + late                                                               # uneaten usual residue, all of it left
            iy_next = iy_c[:, min(t, T - 1)]
            if end.shape[-1] == 2 and np.isfinite(store_cap):
                # bale only for need: the baled part carried out is at most store_cap years of the additional
                # residue the herds ate (DM, all crops); every crop's baled part is cut alike, the rest is lost
                baled = (end[..., 0] * iy_next[..., 0] / crop_me).sum(axis=1)                  # (C,) as carried
                need = store_cap * (fed_extra / crop_me).sum(axis=1)
                cut = np.where(baled > need, need / np.maximum(baled, 1e-30), 1.0)
                lost = end[..., 0] * iy_next[..., 0] * (1.0 - cut[:, None])                   # (C,K) ME
                end[..., 0] *= cut[:, None]
                res["fate"]["store_lost"][:, t - 1] = lost.sum(axis=1)
                res["fate"]["store_lost_dm"][:, t - 1] = (lost / crop_me).sum(axis=1)
            stock_dm = end / crop_me[..., None]
            carry_me = (end * iy_next).sum(axis=-1)
            if end.shape[-1] == 2:
                res["fate"]["carried_baled_dm"][:, t - 1] = (end[..., 0] * iy_next[..., 0] / crop_me).sum(axis=1)
            res["fate"]["carried"][:, t - 1] = carry_me.sum(axis=1)
            res["fate"]["decayed"][:, t - 1] = (left - carry_me + release_lost).sum(axis=1)
            res["fate"]["carried_dm"][:, t-1] = (carry_me/crop_me).sum(1)
            res["fate"]["decayed_dm"][:, t-1] = ((left-carry_me+release_lost)/crop_me).sum(1)
            res["fate"]["fed_extra_dm"][:, t-1] = (fed_extra/crop_me).sum(1)
        if fixed and not sol["fixed_feasible"].all():
            bad = np.argwhere(~sol["fixed_feasible"])
            labels = [f"{prep.iso3[c]}/{prep.herds[h]}" for c, h in bad[:8]]
            raise ValueError(f"Fixed herd or output floor infeasible in year {t}: {', '.join(labels)}")
        n, p = sol["n"], sol["p"]
        res["diet_used"][:, :, t - 1] = sol["used"]
        res["diet_dm"][:, :, t - 1] = sol["dm"]
        res["treated_dm"][:, :, t - 1] = sol["treated_dm"]
        res["diet_md"][:, :, t - 1] = sol["md"]
        res["diet_settled"][:, :, t - 1] = sol["diet_settled"]
        res["urea"][:, :, t - 1] = sol["urea"]
        rel = np.where(prep.n0 > 0, n * p / np.maximum(prep.n0, 1e-9), 0.0)
        res["n"][:, :, t - 1], res["p"][:, :, t - 1] = n, p
        res["binding"][:, :, t - 1] = sol["binding"]
        res["milk"][:, :, t - 1] = prep.milk0 * rel
        meat = prep.meat0 * rel
        if herd_dynamics:
            grown = np.maximum(n - n_prev, 0.0)
            culled = np.maximum(n_prev - n, 0.0)
            meat = np.maximum(meat - grown * prep.carcass_kcal, 0.0)       # animals kept to rebuild are not eaten
            res["carcass"][:, :, t - 1] = culled * prep.carcass_kcal
            res["cull"][:, :, t - 1] = culled
        res["meat"][:, :, t - 1] = meat
        res["conc_used"][:, :, t - 1] = sol["used"][..., CONC]
        res["cop_used"][:, :, t - 1] = sol["used"][..., COP]
        res["grass_used"][:, :, t - 1] = sol["used"][..., GRASS]
        res["res_crop_used"][:, :, t - 1, :] = sol["used"][..., RES_SLICE]
        res["res_used"][:, :, t - 1] = sol["used"][..., RES_SLICE].sum(axis=2)
        n_prev = n
    return res


def food_saved(prep: Prepared, ni: dict, iv: dict):
    """The three components, kcal per year (C,T), intervention minus none.
    Grain is the grain and the bran; the co-products are not human food and
    are not counted."""
    freed_mj = ni["conc_used"] - iv["conc_used"]                                             # (C,H,T)
    return {
        "grain": (freed_mj * prep.grain_kcal_per_mj[:, :, None]).sum(axis=1),
        "milk": (iv["milk"] - ni["milk"]).sum(axis=1),
        "meat": (iv["meat"] - ni["meat"]).sum(axis=1),
        "carcass": (iv["carcass"] - ni["carcass"]).sum(axis=1),
    }
