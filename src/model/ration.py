"""Exact rations: every herd's ration, and a free herd's size, as linear programs.

The conditions on a ration (energy, fiber, protein; caps on any urea supplement) are linear
in the feeds eaten, except that the efficiencies of ME use depend on the ME
content of the diet (M/D, MJ per kg DM). For an assumed content
q they are constants, and the added condition that the diet's actual content
is at least q makes every solution truly feasible, since a richer diet is
used at least as efficiently. The best ration is therefore the best over q.

Each herd's problem is solved with HiGHS (the simplex solver shipped with
scipy), warm-started from one trial content to the next. The duals of the
energy and content rows give the slope of the optimum in q. A coarse scan
over the M/D range finds the region of the best content, and a bisection on
the sign of the slope narrows it to ``Q_TOL``. A trial that is infeasible
(fixed heads) or in which a free herd keeps no animals (the optimum is flat
there) says nothing about the slope: the search moves toward the best point
found. Fixed herds start from the ration they ate without the intervention,
which is feasible at its own content; a start that misses a row or bound of
the problem (beyond ``SEED_TOL``) is dropped. Every ration returned is checked
against energy, MP and NDF apart from the solver (``nutrients_ok``).

Quantities are per reference head per day: the heads available to a free
herd, the fixed heads otherwise. A free herd's share of its heads kept, nu,
is a variable, since every condition scales with heads; P = nu p is the
production of the reference heads. The solver's tolerances are absolute, so a
free herd that keeps a tiny share of its heads and misses its LP is solved
again with twice the heads kept as reference heads. Two auxiliary columns hold the diet's
energy E and dry matter D, so that a trial content changes three
coefficients: q D - E <= 0 (content), -E + nu A(q) + P B(q) <= 0 (energy).

Later aims are small costs, each too small to be bought with an earlier one:
production after grain (fixed herds), then the urea used (supplement and
treatment), then more heads among equal outputs (free herds), then the feed
order of herds fed as today (co-products, grass, grain and bran, residues by
ME per kg NDF).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize._highspy import _core as _hc          # HiGHS as shipped with the pinned scipy
except ImportError:                                            # pragma: no cover
    import highspy as _hc
_Highs = getattr(_hc, "_Highs", None) or getattr(_hc, "Highs")

QLO, QHI = 4.5, 13.5      # M/D range searched for the diet's content
GE_DM = 18.45             # MJ gross energy per kg DM (IPCC 2019, Eq. 10.16 default)
ME_DE = 0.82              # ME per unit DE (NRC 2001)
DE_MIN = 45.0             # DE, % of GE: lowest of IPCC 2019 Table 10.2 (low-quality forage, 45-55%)
Q_EFF_LO = DE_MIN / 100 * GE_DM * ME_DE   # M/D below which the efficiencies are held at their DE_MIN values
Q_TOL = 1e-4              # MJ per kg DM: width at which the search over q stops (the best trial is kept)
STEP = 1.0                # MJ per kg DM: spacing of the coarse scan
# A reference ration solved per head available misses its rows by the solver's tolerance over the share of heads
# kept: up to 1e-6 of a row's terms at 47 Tg (MP, 0.5% of a herd kept); seeds without the reference mixture or urea
# missed by 6e-5 or more.
SEED_TOL = 1e-5           # a seed's worst violation of its LP, relative to each row's terms (Batch.violation)
NUTRIENT_TOL = 2 * SEED_TOL   # a returned ration's energy, MP or NDF shortfall, relative to its terms: an accepted seed passes
# HiGHS's feasibility tolerance is absolute: a free herd keeping a share nu of its heads has rows of size about nu
# and may miss them by 1e-9 / nu of their terms (5e-5 at 150 Tg, under 4e-9 for herds keeping 0.1% or more). Such a
# ration would fail as a seed; it is solved again with the reference heads at twice the heads kept (Batch.solve).
RESCALE_TOL = SEED_TOL / 100  # a free ration's worst violation of its LP above which it is solved again at its own scale
# The later aims, relative to the first. Each must lose to the one above it (a unit of production
# costs several kg of grain; a share of heads costs a third of production; urea buys production when
# protein binds) and still give plateaus of the one above a slope the search can follow.
SECOND = 0.1              # production after grain, per unit of production
UREA_TIE = 1e-2           # urea used, per kg: supplement and treatment
HEAD_TIE = 1e-3           # a free herd's heads, per share kept, among equal outputs
TIER_TIE = 1e-5           # feed order, per MJ eaten and step of the order (at most 24 steps)
APPLICATION = 0.05        # kg urea applied per kg straw DM treated
UREA_PER_KG_BW = 0.0003   # EFSA: kg urea per kg live weight per day
MAX_ADDED_N = 0.30        # share of dietary N from supplemental urea (protein.MAX_ADDED_N_SHARE)
UREA_CP = 0.46 * 6.25


def _de_percent(md):
    """The diet's DE as a percentage of its gross energy, from its M/D, within the range the IPCC ratios are used."""
    return 100.0 * np.clip(md, Q_EFF_LO, QHI) / (ME_DE * GE_DM)


def _rem_reg(de):
    """IPCC 2019 Eqs. 10.14 and 10.15 (Gibbs and Johnson 1993): net energy for maintenance and for growth
    per unit DE, and their slopes in DE%."""
    rem = 1.123 - 4.092e-3 * de + 1.126e-5 * de ** 2 - 25.4 / de
    reg = 1.164 - 5.16e-3 * de + 1.308e-5 * de ** 2 - 37.4 / de
    drem = -4.092e-3 + 2 * 1.126e-5 * de + 25.4 / de ** 2
    dreg = -5.16e-3 + 2 * 1.308e-5 * de + 37.4 / de ** 2
    return rem, reg, drem, dreg


SYSTEMS = ("ipcc", "arc")


def _efficiencies(md, system):
    """(km, kl, kg) and their slopes in M/D."""
    if system == "ipcc":
        rem, reg, drem, dreg = _rem_reg(_de_percent(md))
        inside = (md >= Q_EFF_LO) & (md < QHI)                       # right-hand slope at the lower clip
        dde = 100.0 / (ME_DE * GE_DM)                                 # DE% per MJ/kg of M/D
        dkm, dkg = np.where(inside, drem * dde / ME_DE, 0.0), np.where(inside, dreg * dde / ME_DE, 0.0)
        return rem / ME_DE, rem / ME_DE, reg / ME_DE, dkm, dkm, dkg
    if system == "arc":
        q = np.clip(md, QLO, QHI)
        inside = (md >= QLO) & (md < QHI)
        return 0.02 * q + 0.5, 0.02 * q + 0.4, 0.043 * q, np.where(inside, 0.02, 0.0), np.where(inside, 0.02, 0.0), np.where(inside, 0.043, 0.0)
    raise ValueError(f"efficiency system must be one of {SYSTEMS}")


def efficiencies(md, system="ipcc"):
    """Efficiencies of ME use for maintenance, lactation and gain from the diet's M/D. ``ipcc`` (central):
    IPCC 2019 (Eq. 10.16), maintenance, activity, lactation and pregnancy at REM, growth at REG, both per
    unit DE, turned into ME with ME = 0.82 DE, held at their values for a DE of 45% of gross energy below
    it. ``arc`` (a switch): the ARC (1980) and SCA (1990) efficiencies as given by Corbett and Freer (2003),
    0.02 M/D + 0.5, 0.02 M/D + 0.4 and 0.043 M/D."""
    return _efficiencies(md, system)[:3]


def requirement(nem, nel, neg, nep, q, system="ipcc"):
    """ME for maintenance (A) and for full production (B) at diet content q, and their slopes in q."""
    km, kl, kg, dkm, dkl, dkg = _efficiencies(q, system)
    A = nem / km
    B = nel / kl + neg / kg + nep / km
    dA = -dkm * nem / km ** 2
    dB = -dkl * nel / kl ** 2 - dkg * neg / kg ** 2 - dkm * nep / km ** 2
    return A, B, dA, dB


class Batch:
    """B ration problems with J feed columns.

    Feed columns carry ME ``me``, NDF ``ndf`` (kg/kg DM), MP supply ``mp`` (B,L,J;
    ``mp_urea`` (L,) per kg urea), the room for supplemental urea N ``nb``
    (0.3 CP; retained treatment N does not count), bounds ``lb``/``ub`` (kg DM per reference head per
    day), a feed-order tier and a treated flag. ``pairs`` lists (treated,
    untreated) column pairs sharing one bound, the treated column's.
    ``grain`` is the column capped per head at ``g0`` (kg DM per head kept);
    the co-products are capped only by the herd's supply, which the animals
    left share. ``free``: heads are a variable; otherwise the reference heads are all kept and production is at least ``floor``. ``objective``:
    'output' (most production), 'food' (most milk and meat kcal plus grain
    kcal freed), 'grain' (least grain, then most production) or 'survival'
    (most heads). ``proportional``: output proportional to feed, protein
    ignored (the aggregate-model switch). ``prev``/``carcass`` (free herds,
    'output' or 'food'; strategy 'eaten'): last year's heads as a share of the
    reference heads, and the carcass kcal per head per day; each head kept
    beyond last year's herd then costs its carcass in the objective.
    """

    PER_PROBLEM = ("me", "ndf", "mp", "nb", "lb", "ub", "tier", "treated", "g0", "nem", "nel", "neg", "nep",
                   "mpm", "mpp", "cap", "lw", "pcap", "floor", "outk", "gk", "prev", "carcass")

    def __init__(self, *, me, ndf, mp, mp_urea, nb, lb, ub, tier, treated, pairs, grain, g0,
                 nem, nel, neg, nep, mpm, mpp, cap, lw, pcap, free, floor, objective, outk, gk,
                 proportional=False, urea_max_dm=0.0, use_nb=False, prev=None, carcass=None, system="ipcc"):
        if objective not in ("output", "food", "grain", "survival"):
            raise ValueError("objective must be output, food, grain or survival")
        if (prev is None) != (carcass is None) or (prev is not None and not (free and objective in ("output", "food"))):
            raise ValueError("a growth cost needs prev and carcass, free heads and an output or food objective")
        if system not in SYSTEMS:
            raise ValueError(f"efficiency system must be one of {SYSTEMS}")
        self.prev, self.carcass, self.system = prev, carcass, system
        self.me, self.ndf, self.mp, self.mp_urea, self.nb = me, ndf, mp, np.asarray(mp_urea, float), nb
        self.lb, self.ub, self.tier, self.treated, self.pairs = lb, ub, tier, treated, list(pairs)
        self.grain, self.g0 = grain, g0
        self.nem, self.nel, self.neg, self.nep = nem, nel, neg, nep
        self.mpm, self.mpp, self.cap, self.lw, self.pcap = mpm, mpp, cap, lw, pcap
        self.free, self.floor, self.objective = free, floor, objective
        self.outk = np.where(outk > 0, outk, 1.0)                  # a herd with no recorded product is still run for output
        self.gk = gk
        self.proportional, self.urea_max_dm = proportional, float(urea_max_dm)
        self.use_nb = use_nb or self.urea_max_dm > 0
        self.B, self.J = me.shape
        self.L = mp.shape[1]

    def subset(self, rows):
        """The same problems restricted to ``rows``."""
        sub = Batch.__new__(Batch)
        sub.__dict__.update(self.__dict__)
        for k in self.PER_PROBLEM:
            v = getattr(self, k)
            if v is not None:
                setattr(sub, k, v[rows])
        sub.B = len(rows)
        return sub

    # ---- one problem's LP
    def layout(self):
        """Column positions: feeds, [urea], heads, production, [growth], the diet's energy and dry matter."""
        J = self.J
        col = {"feeds": slice(0, J)}
        k = J
        if self.urea_max_dm > 0:
            col["z"] = k; k += 1
        col["nu"], col["P"] = k, k + 1
        k += 2
        if self.prev is not None:                                       # before E: the stage-2 seed keeps it
            col["g"] = k; k += 1
        col["E"], col["D"] = k, k + 1
        return col, k + 2

    def _scale(self, i):
        if self.objective == "food":
            return max(self.outk[i], self.gk[i], 1e-12)
        if self.objective == "grain":
            return self.gk[i] if self.gk[i] > 0 else 1.0
        return 1.0

    def cost(self, i, stage2=False):
        """The objective of problem i, divided by its scale: the first aim is of order one."""
        col, n = self.layout()
        F, NU, P = col["feeds"], col["nu"], col["P"]
        c = np.zeros(n)
        s = self._scale(i)
        if self.objective == "output":
            c[P] = -1.0
        elif self.objective == "food":
            c[P] = -self.outk[i] / s; c[self.grain] = self.gk[i] / s
        elif self.objective == "grain":
            c[self.grain] = self.gk[i] / s; c[P] = -SECOND
        else:
            c[NU] = -1.0; c[P] = -SECOND
        if "g" in col:
            # a head kept beyond last year's herd is a carcass not eaten this year, in the units of c[P]; the
            # reported meat is clipped at zero (mechanism.animal_output), this cost is not: the LP stays linear
            c[col["g"]] = self.carcass[i] / (self.outk[i] if self.objective == "output" else s)
        if stage2:                                                # the largest herd first, then the herd's own aim
            c = SECOND * c
            c[NU] -= 1.0
        elif self.free and self.objective in ("output", "food"):
            c[NU] -= HEAD_TIE                                        # among equal outputs, more heads
        c[F] += TIER_TIE * self.tier[i] * self.me[i]                 # feed order, per MJ
        c[F] += UREA_TIE * APPLICATION * self.treated[i]
        if "z" in col:
            c[col["z"]] = UREA_TIE
        return c

    def worth(self, i, x):
        """What a free herd maximises, per reference head per day."""
        col, _ = self.layout()
        if self.objective == "food":
            w = x[col["P"]] * self.outk[i] + (self.ub[i, self.grain] - x[self.grain]) * self.gk[i]
            return w - x[col["g"]] * self.carcass[i] if "g" in col else w
        return x[col["P"]] - x[col["g"]] * self.carcass[i] / self.outk[i] if "g" in col else x[col["P"]]

    def lp(self, i, q, floor=None):
        """Problem i's LP at content q: constraint matrix, row bounds, column bounds and cost; ``floor``:
        least worth (the largest-herd stage)."""
        col, n = self.layout()
        F, NU, P, E, D = col["feeds"], col["nu"], col["P"], col["E"], col["D"]
        Z = col.get("z")
        A_, B_, _, _ = requirement(self.nem[i], self.nel[i], self.neg[i], self.nep[i], q, self.system)
        me = self.me[i]
        rows, lo_r, up_r = [], [], []

        def row(entries, lo=-np.inf, up=0.0):
            r = np.zeros(n)
            for j, v in entries:
                r[j] += v
            rows.append(r); lo_r.append(lo); up_r.append(up)

        # 0 energy, 1 content: the rows whose coefficients follow q
        if self.proportional:
            row([(E, -1.0), (P, A_ + B_)])
        else:
            row([(E, -1.0), (NU, A_), (P, B_)])
        row([(D, q), (E, -1.0)])
        r = np.zeros(n); r[E] = 1.0; r[F] = -me; rows.append(r); lo_r.append(0.0); up_r.append(0.0)   # E = sum m x
        r = np.zeros(n); r[D] = 1.0; r[F] = -1.0
        if Z is not None:
            r[Z] = -1.0
        rows.append(r); lo_r.append(0.0); up_r.append(0.0)                                          # D = sum x (+ urea)
        r = np.zeros(n); r[F] = self.ndf[i]; r[NU] = -self.cap[i]; rows.append(r); lo_r.append(-np.inf); up_r.append(0.0)
        if not self.proportional:
            for l in range(self.L):
                r = np.zeros(n); r[F] = -self.mp[i, l]
                if Z is not None:
                    r[Z] = -self.mp_urea[l]
                r[NU] = self.mpm[i]; r[P] = self.mpp[i]
                rows.append(r); lo_r.append(-np.inf); up_r.append(0.0)
        row([(P, 1.0), (NU, -min(self.pcap[i], 1.0))])                 # production at most full
        row([(self.grain, 1.0), (NU, -self.g0[i])])                     # grain and bran per head
        if "g" in col:                                                  # growth: the heads beyond last year's
            row([(NU, 1.0), (col["g"], -1.0)], up=self.prev[i])
        if self.use_nb:
            r = np.zeros(n); r[F] = -self.nb[i]
            if Z is not None:
                r[Z] = (1 - MAX_ADDED_N) * UREA_CP                     # supplemental urea N at most 30% of dietary N
            rows.append(r); lo_r.append(-np.inf); up_r.append(0.0)
        if Z is not None:
            a = self.urea_max_dm
            r = np.zeros(n); r[Z] = 1 - a; r[F] = -a; rows.append(r); lo_r.append(-np.inf); up_r.append(0.0)
            row([(Z, 1.0), (NU, -UREA_PER_KG_BW * self.lw[i])])
        for jt, ju in self.pairs:                                       # treated and untreated share the straw
            row([(jt, 1.0), (ju, 1.0)], up=self.ub[i, jt])
        if floor is not None:                                           # worth at least the floor
            gc = [(col["g"], self.carcass[i])] if "g" in col else []   # the growth cost, kcal per reference head per day
            if self.objective == "food":
                row([(P, -self.outk[i]), (self.grain, self.gk[i])] + gc, up=self.gk[i] * self.ub[i, self.grain] - floor)
            else:
                row([(P, -1.0)] + [(j, v / self.outk[i]) for j, v in gc], up=-floor)
        Am = np.array(rows)
        lo_c, up_c = np.zeros(n), np.full(n, np.inf)
        lo_c[F], up_c[F] = self.lb[i], np.where(self.ub[i] < 1e-12, 0.0, self.ub[i])
        lo_c[F] = np.minimum(lo_c[F], up_c[F])
        if self.free:
            up_c[NU], up_c[P] = 1.0, 1.0
        else:
            lo_c[NU] = up_c[NU] = 1.0
            up_c[P] = min(self.pcap[i], 1.0)
            lo_c[P] = min(self.floor[i], up_c[P])
        return Am, np.array(lo_r, float), np.array(up_r, float), lo_c, up_c, self.cost(i, stage2=floor is not None)

    def model(self, i, q, floor=None):
        """A HiGHS model of problem i at content q; ``floor``: least worth (the largest-herd stage)."""
        Am, lo_r, up_r, lo_c, up_c, cost = self.lp(i, q, floor=floor)
        n = Am.shape[1]
        lp = _hc.HighsLp()
        lp.num_col_, lp.num_row_ = n, Am.shape[0]
        lp.col_cost_ = cost
        lp.col_lower_, lp.col_upper_ = lo_c, up_c
        lp.row_lower_, lp.row_upper_ = lo_r, up_r
        nz = Am != 0
        lp.a_matrix_.format_ = _hc.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.concatenate([[0], np.cumsum(nz.sum(0))]).astype(np.int32)
        cols_nz = [np.flatnonzero(nz[:, j]) for j in range(n)]
        lp.a_matrix_.index_ = np.concatenate(cols_nz).astype(np.int32) if cols_nz else np.zeros(0, np.int32)
        lp.a_matrix_.value_ = np.concatenate([Am[idx, j] for j, idx in enumerate(cols_nz)]).astype(float)
        h = _Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("presolve", "off")                              # keep the basis from one trial to the next
        h.setOptionValue("primal_feasibility_tolerance", 1e-9)
        h.setOptionValue("dual_feasibility_tolerance", 1e-9)
        h.passModel(lp)
        return h

    def violation(self, i, x, q, floor=None):
        """Worst violation by x of problem i's rows and column bounds at content q: a row's relative to the
        size of its terms and bound, a column's to its value and bound, at least one."""
        Am, lo_r, up_r, lo_c, up_c, _ = self.lp(i, q, floor=floor)
        ax = Am @ x
        fin = lambda b: np.where(np.isfinite(b), np.abs(b), 0.0)
        size = np.abs(Am) @ np.abs(x) + np.maximum(fin(lo_r), fin(up_r))
        rows = np.maximum(lo_r - ax, ax - up_r) / np.maximum(size, 1e-30)
        cols = np.maximum(lo_c - x, x - up_c) / np.maximum(np.maximum(np.abs(x), np.maximum(fin(lo_c), fin(up_c))), 1.0)
        return float(np.max(np.concatenate([rows, cols])))              # nan (a bad content): infeasible

    def returned_miss(self, i, x, q):
        """Batch.violation of solution x of problem i at content q with its feeds clipped to their bounds and the
        diet's energy and dry matter recomputed, as ``unpack`` returns it: the solver's tolerance on a bound is
        absolute, and on a feed offered in tiny amounts the clipped excess can hold much of the ration's energy."""
        col, _ = self.layout()
        F = col["feeds"]
        xc = np.array(x, float)
        xc[F] = np.clip(xc[F], 0.0, np.where(self.ub[i] < 1e-12, 0.0, self.ub[i]))
        z = max(xc[col["z"]], 0.0) if "z" in col else 0.0
        if "z" in col:
            xc[col["z"]] = z
        xc[col["E"]] = xc[F] @ self.me[i]
        xc[col["D"]] = xc[F].sum() + z
        return self.violation(i, xc, q)

    def probe(self, h, i, q):
        """Re-solve problem i at content q: (feasible, x, value, slope)."""
        col, _ = self.layout()
        A_, B_, dA, dB = requirement(self.nem[i], self.nel[i], self.neg[i], self.nep[i], q, self.system)
        if self.proportional:
            h.changeCoeff(0, col["P"], float(A_ + B_))
        else:
            h.changeCoeff(0, col["nu"], float(A_)); h.changeCoeff(0, col["P"], float(B_))
        h.changeCoeff(1, col["D"], float(q))
        h.run()
        if h.getModelStatus() != _hc.HighsModelStatus.kOptimal:
            return False, None, np.inf, 0.0
        sol = h.getSolution()
        x = np.asarray(sol.col_value)
        lam = -np.asarray(sol.row_dual[:2])                              # rise of the optimum per unit tightening
        value = h.getInfo().objective_function_value
        if self.proportional:
            slope = lam[0] * x[col["P"]] * (dA + dB) + lam[1] * x[col["D"]]
        else:
            slope = lam[0] * (x[col["nu"]] * dA + x[col["P"]] * dB) + lam[1] * x[col["D"]]
        return True, x, value, slope

    # ---- the search over q
    def search_one(self, i, tol=Q_TOL, anchor=None, seed=None, floor=None, hint=None, step=STEP):
        """Best feasible solution of problem i over q: coarse scan (or a bracket around ``hint``, the
        content found by an earlier, similar solve, if it holds the optimum), then bisection on the
        slope's sign."""
        col, n = self.layout()
        h = self.model(i, 9.0, floor=floor)
        best_x, best_v, best_q = None, np.inf, np.nan
        if seed is not None:
            x0 = np.zeros(n); x0[:len(seed[0])] = seed[0]
            F = col["feeds"]
            x0[col["E"]] = x0[F] @ self.me[i]
            x0[col["D"]] = x0[F].sum() + (x0[col["z"]] if "z" in col else 0.0)
            if self.violation(i, x0, float(seed[1]), floor=floor) <= SEED_TOL:   # an infeasible seed is dropped
                best_x, best_v, best_q = x0, float(self.cost(i, stage2=floor is not None) @ x0), float(seed[1])
        n_lp = 0

        def informative(ok, x):
            return ok and (not self.free or x[col["nu"]] > 1e-9)

        lo = hi = centre = None
        if hint is not None and np.isfinite(hint):
            # a narrow bracket around the hint, kept if the slope says the optimum is inside
            ok, x, v, s = self.probe(h, i, hint); n_lp += 1
            if informative(ok, x):
                if v < best_v:
                    best_x, best_v, best_q = x, v, hint
                a, b = max(QLO, hint - 0.25 * step), min(QHI, hint + 0.25 * step)
                ok_a, xa, va, sa = self.probe(h, i, a); ok_b, xb, vb, sb = self.probe(h, i, b); n_lp += 2
                for okk, xx, vv, qq in ((ok_a, xa, va, a), (ok_b, xb, vb, b)):
                    if informative(okk, xx) and vv < best_v:
                        best_x, best_v, best_q = xx, vv, qq
                left = (not informative(ok_a, xa)) or sa < 0 or a <= QLO
                right = (not informative(ok_b, xb)) or sb >= 0 or b >= QHI
                if left and right:
                    lo, hi, centre = a, b, hint
        if lo is None:
            for q in np.arange(QLO, QHI + 1e-9, step):
                ok, x, v, s = self.probe(h, i, q); n_lp += 1
                if informative(ok, x) and v < best_v:
                    best_x, best_v, best_q = x, v, q
            centre = best_q if np.isfinite(best_q) else (anchor if anchor is not None else None)
            if centre is None:                                          # nothing feasible, nothing to anchor on
                return best_x, best_v, best_q, False, n_lp
            lo, hi = max(QLO, centre - step), min(QHI, centre + step)
        while hi - lo > tol:
            q = 0.5 * (lo + hi)
            ok, x, v, s = self.probe(h, i, q); n_lp += 1
            if informative(ok, x):
                if v < best_v:
                    best_x, best_v, best_q = x, v, q
                up = s < 0
            else:
                up = q < centre
            lo, hi = (q, hi) if up else (lo, q)
        # At the optimum the diet's actual M/D equals the assumed one (else raising q would lower the
        # requirement for nothing). From the bracket's lower end, raise q to the diet's actual M/D until it
        # holds: each step keeps the diet feasible, so the optimum can only improve.
        q = lo
        for _ in range(30):
            ok, x, v, s = self.probe(h, i, q); n_lp += 1
            if not informative(ok, x):
                break
            if v < best_v:
                best_x, best_v, best_q = x, v, q
            md = x[col["E"]] / max(x[col["D"]], 1e-30)
            if md <= q + 1e-9 or md >= QHI:
                break
            q = md
        return best_x, best_v, best_q, best_x is not None, n_lp

    def solve(self, tol=Q_TOL, within=0.005, anchor=None, seed=None, hint=None, step=STEP):
        """Every problem's ration; free herds with an 'output' or 'food' objective keep the largest herd whose
        worth is within ``within`` of the best. ``anchor``/``seed``: a feasible content and solution for
        fixed heads (the ration eaten without the intervention)."""
        col, n = self.layout()
        X = np.zeros((self.B, n)); Q = np.full(self.B, np.nan); conv = np.zeros(self.B, bool)
        n_lp = 0
        for i in range(self.B):
            a = None if (self.free or anchor is None) else float(anchor[i])
            sd = None if (self.free or seed is None) else (seed[0][i], seed[1][i])
            x, q, ok, k = self._solve_one(i, tol, within, a, sd, None if hint is None else hint[i], step)
            n_lp += k
            if x is None:
                continue
            nu = x[col["nu"]]
            if self.free and 0 < nu < 0.5 and (miss := self.returned_miss(i, x, q)) > RESCALE_TOL:
                # a tiny share of heads kept: solved again per twice the heads kept, where the solver's absolute
                # tolerances are small beside the rows' terms and the feeds' bounds; every column scales with the
                # reference heads
                s = 2.0 * nu
                sub = self.subset([i])
                sub.lb, sub.ub = sub.lb / s, sub.ub / s
                if sub.prev is not None:
                    sub.prev = sub.prev / s
                x3, q3, ok3, k3 = sub._solve_one(0, tol, within, None, None, q, step)
                n_lp += k3
                if x3 is not None and ok3 and self.returned_miss(i, x3 * s, q3) < miss:
                    x, q, ok = x3 * s, q3, ok3
            X[i], Q[i], conv[i] = x, q, ok
        # a free herd that can keep no animals at any content is gone: its empty ration is the answer
        if self.free:
            conv[:] = True
        return self.unpack(X, Q, conv, n_lp)

    def _solve_one(self, i, tol, within, anchor, seed, hint, step):
        """Problem i's ration, then for a free herd with an 'output' or 'food' objective the largest herd whose worth
        is within ``within`` of the best: (x, q, converged, LPs solved); x is None if nothing is feasible."""
        col, _ = self.layout()
        x, v, q, ok, n_lp = self.search_one(i, tol=tol, anchor=anchor, seed=seed, hint=hint, step=step)
        if x is None:
            return None, q, ok, n_lp
        if self.free and self.objective in ("output", "food") and within is not None and x[col["nu"]] < 1 - 1e-9:
            best = self.worth(i, x)
            # the stage-1 ration is feasible for stage 2: it seeds and anchors the search
            x2, v2, q2, ok2, k2 = self.search_one(i, tol=tol, floor=best - within * abs(best), hint=q,
                                                  anchor=q, seed=(x[:col["E"]], q), step=step)
            n_lp += k2
            if ok2 and x2[col["nu"]] >= x[col["nu"]] - 1e-12:
                x, q = x2, q2
        return x, q, ok, n_lp

    def unpack(self, X, Q, conv, n_lp):
        col, _ = self.layout()
        F = col["feeds"]
        feeds = np.clip(X[:, F], 0.0, np.where(np.isfinite(self.ub), self.ub, np.inf))   # solver rounding
        z = np.maximum(X[:, col["z"]], 0.0) if "z" in col else np.zeros(self.B)
        nu, P = X[:, col["nu"]].copy(), X[:, col["P"]].copy()
        full = nu >= 1 - 1e-9                                           # solver rounding: all heads kept
        P = np.where(full, P / np.maximum(nu, 1e-30), P)
        nu = np.where(full, 1.0, nu)
        E = (feeds * self.me).sum(1)
        D = feeds.sum(1) + z
        md = np.where(D > 0, E / np.maximum(D, 1e-30), 0.0)
        A_, B_, _, _ = requirement(self.nem, self.nel, self.neg, self.nep, np.where(D > 0, md, QHI), self.system)
        safe = np.maximum(nu, 1e-30)
        if self.proportional:
            p_e = E / safe / (A_ + B_)
            p_mp = np.full(self.B, np.inf)
        else:
            p_e = (E / safe - A_) / B_
            mp = np.einsum("blj,bj->bl", self.mp, feeds) + self.mp_urea[None] * z[:, None]
            p_mp = ((mp / safe[:, None] - self.mpm[:, None]) / self.mpp[:, None]).min(1)
        fiber = (feeds * self.ndf).sum(1)
        room = self.cap * nu - fiber
        p = np.where(nu > 0, np.clip(P / safe, 0.0, 1.0), 0.0)
        p = np.where(p >= 1 - 1e-9, 1.0, p)                             # solver rounding: full production
        # whatever the solver said, the feeds returned support the production returned at the diet's actual
        # M/D (p <= p_e, p <= p_mp) and fit the NDF limit: each shortfall within NUTRIENT_TOL of its terms
        e, need = E / safe, (A_ + B_) * p if self.proportional else A_ + B_ * p
        ok = (need - e <= NUTRIENT_TOL * (need + e)) & (-room <= NUTRIENT_TOL * (self.cap * nu + fiber))
        if not self.proportional:
            supply, need_mp = mp / safe[:, None], (self.mpm + self.mpp * p)[:, None]
            ok &= (need_mp - supply <= NUTRIENT_TOL * (need_mp + supply)).all(1)
        ok |= nu <= 0                                                   # no heads kept
        return dict(feeds=feeds, urea=z, nu=nu, p=p, md=md, q=Q, p_e=p_e, p_mp=p_mp, room=room,
                    feasible=conv.copy(), conv=conv, nutrients_ok=ok, n_lp=n_lp)
