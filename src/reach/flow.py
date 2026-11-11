"""Maximum flow of residue energy from source cells to animal cells within a radius.

Sources i (residue ME e_i) and destinations j (ruminant ME demand d_j) are the
cells of one country on the 5 arcmin grid. An arc joins i and j when their
centres are at most R km apart on the sphere. The allocation maximizes the
energy moved,

    max sum_ij x_ij   s.t.   sum_j x_ij <= tau e_i,   sum_i x_ij <= mu d_j,

with x_ij = 0 off the arcs. Scaling every capacity by one factor scales the
optimum by that factor, so F(tau, mu) = tau G(mu / tau) where G(r) is the
optimum with the whole supply removable and the demand scaled by r. G is
nondecreasing, concave and piecewise linear in r; a chord between two of its
values is the energy of a feasible flow (a mixture of the two optimal flows),
so linear interpolation between tabulated ratios never claims more than the
network can carry.

The solver is scipy's Dinic implementation on integer capacities. Capacities
are expressed in units of max(sum e, r_max sum d) / 2^30, so every capacity
fits in int32 and each cell is rounded by at most half a unit, about 1e-9 of
the country total; the unit is reported with the results. Successive ratios
reuse the previous flow: the next solve runs on the residual network (the
supply left, reverse arcs carrying the flow already placed, the enlarged
demand), whose optimum is the increment. Every flow is checked against both
capacities, for conservation and for maximality (no augmenting path left in
its residual network) before it is accepted; a sweep that fails a check is
solved again from zero flow (RETRIES), since the solver is deterministic and
a failure means the computation was corrupted.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

EARTH_KM = 6371.0088
CELL_DEG = 1 / 12           # 5 arcmin
FULL_WIDTH = 4320           # columns of the global grid; a box this wide wraps at the date line
ARC_CAP = 2 ** 30           # an arc never limits: the source capacity does
SCALE = 2 ** 30             # the largest capacity in integer units
RETRIES = 2                 # a sweep that fails a check is solved again from zero flow this many times


def reach_edges(src_rows, src_cols, dst_rows, dst_cols, radius_km, lat_centres, ncol, *, cell_deg=CELL_DEG, wrap=None):
    """Index pairs (u into the sources, v into the destinations) of cells whose centres are
    at most radius_km apart on the sphere. Rows are grid rows with latitudes lat_centres
    (degrees, one per row of the box) and columns are cell_deg wide; a box that spans the
    globe (wrap, by default a FULL_WIDTH box at 5 arcmin) wraps at the date line.

    For a source row and a destination row the reachable columns form an interval
    (the cap of radius R around a source meets each row in one arc), so the arcs of each row
    pair are found by two binary searches per source."""
    src_rows, src_cols = np.asarray(src_rows), np.asarray(src_cols)
    dst_rows, dst_cols = np.asarray(dst_rows), np.asarray(dst_cols)
    lat = np.deg2rad(np.asarray(lat_centres, float))
    if radius_km < 0 or not np.isfinite(radius_km):
        raise ValueError("The radius must be finite and nonnegative")
    reach = np.sin(min(radius_km / EARTH_KM, np.pi) / 2) ** 2
    if wrap is None:
        wrap = ncol * cell_deg >= 360 - 1e-9
    order = np.argsort(dst_rows, kind="stable")
    d_rows, d_cols = dst_rows[order], dst_cols[order]
    start = np.searchsorted(d_rows, np.arange(len(lat) + 1))
    # destination columns sorted within each row, with their original indices
    cols_by_row, index_by_row = {}, {}
    for j in np.unique(d_rows):
        seg = slice(start[j], start[j + 1])
        o = np.argsort(d_cols[seg], kind="stable")
        cols_by_row[j] = d_cols[seg][o]
        index_by_row[j] = order[seg][o]
    us, vs = [], []
    for i in np.unique(src_rows):
        si = np.flatnonzero(src_rows == i)
        cs = src_cols[si]
        rows = np.flatnonzero(np.abs(lat - lat[i]) <= radius_km / EARTH_KM + 1e-14)
        rows = rows[start[rows + 1] > start[rows]]
        if len(rows) == 0:
            continue
        h = (reach - np.sin((lat[i] - lat[rows]) / 2) ** 2) / (np.cos(lat[i]) * np.cos(lat[rows]))
        widths = np.floor(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))) / np.deg2rad(cell_deg) + 1e-10).astype(int)
        for j, w in zip(rows, widths):
            cd, idx = cols_by_row[j], index_by_row[j]
            if wrap and w >= ncol // 2:
                lo, hi = np.zeros(len(cs), int), np.full(len(cs), len(cd))
            elif wrap:
                cd = np.concatenate([cd - ncol, cd, cd + ncol])
                idx = np.concatenate([idx, idx, idx])
                lo, hi = np.searchsorted(cd, cs - w, side="left"), np.searchsorted(cd, cs + w, side="right")
            else:
                lo, hi = np.searchsorted(cd, cs - w, side="left"), np.searchsorted(cd, cs + w, side="right")
            n = hi - lo
            total = int(n.sum())
            if total == 0:
                continue
            us.append(np.repeat(si, n))
            vs.append(idx[np.repeat(lo, n) + (np.arange(total) - np.repeat(np.cumsum(n) - n, n))])
    if not us:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(us).astype(np.int64), np.concatenate(vs).astype(np.int64)


class ReachNetwork:
    """The bipartite network of one country box at one radius.

    supply and demand are matching 2-D arrays (GJ); lat_centres has one latitude per row and
    cell_deg is the width of a column. Cells with zero supply or zero demand are not nodes;
    nothing is rounded away here.
    """

    def __init__(self, supply, demand, radius_km, lat_centres, *, cell_deg=CELL_DEG, wrap=None):
        supply, demand = np.asarray(supply, float), np.asarray(demand, float)
        if supply.shape != demand.shape or supply.ndim != 2 or len(lat_centres) != supply.shape[0]:
            raise ValueError("Supply and demand must be matching 2-D grids with one latitude per row")
        if not (np.isfinite(supply).all() and np.isfinite(demand).all()) or (supply < 0).any() or (demand < 0).any():
            raise ValueError("Supply and demand must be finite and nonnegative")
        ncol = supply.shape[1]
        self._shape = supply.shape
        self.src = np.flatnonzero(supply.ravel() > 0)
        self.dst = np.flatnonzero(demand.ravel() > 0)
        self.e = supply.ravel()[self.src]
        self.d = demand.ravel()[self.dst]
        self.ns, self.nd = len(self.src), len(self.dst)
        self.u, self.v = reach_edges(self.src // ncol, self.src % ncol, self.dst // ncol, self.dst % ncol,
                                     radius_km, lat_centres, ncol, cell_deg=cell_deg, wrap=wrap)
        has_src = np.zeros(self.ns, bool); has_src[self.u] = True
        has_dst = np.zeros(self.nd, bool); has_dst[self.v] = True
        self.reach_supply = float(self.e[has_src].sum())      # supply with any demand in reach
        self.reach_demand = float(self.d[has_dst].sum())      # demand with any supply in reach
        self.supply = float(self.e.sum())
        self.demand = float(self.d.sum())
        key = self.u * max(self.nd, 1) + self.v
        self._order = np.argsort(key, kind="stable")
        self._keys = key[self._order]

    @property
    def n_edges(self):
        return len(self.u)

    def _graph(self, ce, cd, f_src, f_uv, f_dst):
        """Residual network of the flow (f_src, f_uv, f_dst) for capacities ce (sources) and cd (sinks)."""
        ns, nd = self.ns, self.nd
        S, T = ns + nd, ns + nd + 1
        used = f_uv > 0
        rows = np.concatenate([np.full(ns, S), self.u, ns + self.v[used], ns + np.arange(nd)])
        cols = np.concatenate([np.arange(ns), ns + self.v, self.u[used], np.full(nd, T)])
        data = np.concatenate([ce - f_src, np.full(len(self.u), ARC_CAP, np.int64), f_uv[used], cd - f_dst])
        if (data < 0).any():
            raise AssertionError("Negative residual capacity")
        g = sp.csr_array((data.astype(np.int32), (rows, cols)), shape=(T + 1, T + 1))
        g.sum_duplicates()
        return g, S, T

    def _increment(self, flow, S, T):
        """Flow increments on the sources, the arcs and the sinks from a scipy flow matrix."""
        ns, nd = self.ns, self.nd
        fl = flow.tocoo()
        row, col, val = fl.row, fl.col, fl.data.astype(np.int64)
        m = row == S
        d_src = np.zeros(ns, np.int64); np.add.at(d_src, col[m], val[m])
        m = col == T
        d_dst = np.zeros(nd, np.int64); np.add.at(d_dst, row[m] - ns, val[m])
        m = (row < ns) & (col >= ns) & (col < ns + nd) & (val != 0)
        d_uv = np.zeros(len(self.u), np.int64)
        if m.any():
            keys = row[m].astype(np.int64) * max(nd, 1) + (col[m] - ns)
            pos = self._order[np.searchsorted(self._keys, keys)]
            if not np.array_equal(self._keys[np.searchsorted(self._keys, keys)], keys):
                raise AssertionError("Flow on an arc that is not in the network")
            np.add.at(d_uv, pos, val[m])
        return d_src, d_uv, d_dst

    def curve(self, ratios, allocation_at=()):
        """G(r) (GJ) for increasing demand ratios r, the per-source allocation (GJ, aligned
        with self.src) at the ratios in allocation_at, and the integer unit (GJ). Ratios above the
        one at which the flow first reaches every source with demand in reach are not solved again:
        the optimum stays at reach_supply."""
        ratios = np.asarray(ratios, float)
        if len(ratios) == 0 or (np.diff(ratios) <= 0).any() or ratios[0] <= 0:
            raise ValueError("Ratios must be positive and strictly increasing")
        out = np.zeros(len(ratios))
        allocs = {}
        if self.n_edges == 0:
            for r in allocation_at:
                allocs[r] = np.zeros(self.ns)
            return out, allocs, 0.0
        unit = max(self.supply, ratios[-1] * self.demand) / SCALE
        return self._checked_sweep(ratios, unit, allocation_at, allocs), allocs, unit

    def _checked_sweep(self, *args):
        """_sweep, solved again from zero flow when a check fails (at most RETRIES times, counted in self.retries).
        The solver is deterministic, so a sweep that fails a check and then passes on the same input met a
        corrupted computation, not a property of the network; a sweep that fails every time raises."""
        for attempt in range(RETRIES + 1):
            try:
                return self._sweep(*args)
            except AssertionError as err:
                if attempt == RETRIES:
                    raise
                self.retries = getattr(self, "retries", 0) + 1
                warnings.warn(f"Maximum-flow check failed ({err}); the sweep is solved again from zero flow", RuntimeWarning)

    def _sweep(self, ratios, unit, allocation_at, allocs):
        """G at increasing ratios from zero flow in one integer unit, reusing the flow along the ratios;
        fills allocs."""
        ns, nd = self.ns, self.nd
        out = np.zeros(len(ratios))
        ce = np.rint(self.e / unit).astype(np.int64)
        f_src = np.zeros(ns, np.int64); f_dst = np.zeros(nd, np.int64); f_uv = np.zeros(len(self.u), np.int64)
        has_src = np.zeros(ns, bool); has_src[self.u] = True
        cap_reach = int(ce[has_src].sum())
        for k, r in enumerate(ratios):
            if f_src.sum() < cap_reach:
                cd = np.rint(r * self.d / unit).astype(np.int64)
                if (cd < f_dst).any():
                    raise AssertionError("Demand capacities must not shrink along the sweep")
                g, S, T = self._graph(ce, cd, f_src, f_uv, f_dst)
                res = maximum_flow(g, S, T)
                d_src, d_uv, d_dst = self._increment(res.flow, S, T)
                f_src, f_uv, f_dst = f_src + d_src, f_uv + d_uv, f_dst + d_dst
                if int(res.flow_value) != int(d_src.sum()):
                    raise AssertionError("Flow value and source increments disagree")
                self._check(ce, cd, f_src, f_uv, f_dst)
                self._certify(ce, cd, f_src, f_uv, f_dst)
            out[k] = f_src.sum() * unit
            if r in allocation_at:
                allocs[r] = f_src * unit
        return out

    def _check(self, ce, cd, f_src, f_uv, f_dst):
        if (f_src > ce).any() or (f_dst > cd).any() or (f_uv < 0).any():
            raise AssertionError("A capacity is violated")
        by_src = np.bincount(self.u, weights=f_uv, minlength=self.ns)
        by_dst = np.bincount(self.v, weights=f_uv, minlength=self.nd)
        if not (np.array_equal(by_src, f_src) and np.array_equal(by_dst, f_dst)):
            raise AssertionError("Flow is not conserved")

    def _certify(self, ce, cd, f_src, f_uv, f_dst):
        """The flow is maximal: the sink cannot be reached from the source in its residual network
        (a feasible flow that passes _check could still fall short)."""
        g, S, T = self._graph(ce, cd, f_src, f_uv, f_dst)
        g.eliminate_zeros()
        if (breadth_first_order(g, S, directed=True, return_predecessors=False) == T).any():
            raise AssertionError("The flow is not maximal")

    def solve(self, r=1.0, tau=1.0):
        """Maximum flow (GJ) and per-source allocation for one (tau, r): tau e_i against r d_j."""
        total, allocs, _ = self.curve([r / tau], allocation_at=(r / tau,))
        return tau * total[0], tau * allocs[r / tau]
