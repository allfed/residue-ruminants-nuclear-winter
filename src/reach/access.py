"""Reach: the share of a country's residue energy that ruminants within R km can get, as a maximum flow.

The reach tables are distributed as inputs: data/residue_access.csv (per country, scenario, year,
radius and demand ratio), data/residue_access_crops.npz (each crop alone, on the pre-catastrophe
grid) and the 0.5-degree map arrays data/map_context/maps_05deg.npz. They hold the allocation
described here, computed on the 5 arcmin grid of reach/grid.py. This module puts residue and
demand on that grid and cuts it into country grids; Fig. 2 solves the allocation on them again,
cell by cell.

Residue ME per pixel is SPAM 2020 production per crop times the country's residue ratio
(model/ratios.py, central, at the national yield) and ME per tonne (residue_feed_values.csv;
residue_grid, per_tonne). For scenario tg and year t, each crop's production is scaled by the
country's change in the frozen crop tables, tonnes[c, crop, t] / tonnes[c, crop, 0], and the
ratio is the national ratio of that year (ratios.central(tg)[:, :, t]). No per-cell yield change
is applied: within a country every pixel of a crop moves by the same factor, so the pattern
changes only through the country's crop mix. Ruminant ME demand per pixel is GLW4 heads times
the country's mean requirement per head of the species (ruminant_herds.csv; demand_grids). It
keeps its pre-catastrophe pattern in every scenario and year; its level is the surviving herd's,
set by the national model (below). Year 0 is the pre-catastrophe grid, identical across scenarios.

The allocation is the maximum flow of reach/flow.py: residue x_ij moved
from source cell i to animal cell j of the same country within R km,
maximizing the total subject to sum_j x_ij <= tau e_i (tau is the share
that may leave the field, 1 minus the reserves of residue_reserve.csv) and
sum_i x_ij <= mu d_j (mu is the share of the pre-catastrophe ruminant
energy demand that the surviving herd represents). Scaling both capacities
scales the optimum, so the share a(tau, mu) = tau g(mu / tau), where g(r)
is the allocated share with the whole residue removable and the demand
scaled by r. The table gives g at the ratios RATIOS (columns
access_{R}km_r{r}) for each country, scenario and year, plus the shares of
residue with any demand in reach (reach_supply_{R}km, the limit of g) and
of demand with any residue in reach (reach_demand_{R}km, in residue units,
the slope of g at zero). model/prepare.py interpolates g linearly in r,
which stays feasible (a mixture of two feasible flows), and in log R.
RATIOS run from 1.024e-4 to 20, a factor 1.22 apart; those below SPLIT
(0.05) are solved in a sweep of their own, in a finer integer unit.
Below the first ratio the model scales g[0] down proportionally (a
scaled-down flow), which understates the flow by at most r_0 D_reach / 4
(G concave, G(r) <= r D_reach): 2.6e-5 of the demand in reach, whatever
the curve. The columns interpolation_bound_{R}km give the most that the
chords between tabulated ratios, or this proportional reading, can
understate g.
The grids give the patterns; the model puts the national levels on them
(model/prepare.reach_scale): with national residue S and national eating
capacity Cap on the grid's patterns, homogeneity gives the share
tau g(r / tau) at r = (Cap / S) (residue_pj / demand_pj), so the allocation
never exceeds Cap whatever the grid's totals or the draw's requirements.

Per crop (residue_access_crops.npz): each counted crop's
residue alone flows to the same animal cells, g_k(r) at the same ratios and
radii, on the pre-catastrophe grid only: within a country every cell of a
crop moves by one factor in each scenario, year and draw, so by homogeneity
g_k of year 0 at r = (Cap / S_k) (E_k / D), with E_k the crop's gridded
residue of year 0, covers them all. The model offers no crop more than its
own reach.

Radii of 10, 25 and 50 km are solved on the 5 arcmin cells; 100 and 200 km
on 10 arcmin cells (2 x 2 sums within the country mask, so borders stay
exact: country_grids with f = 2), where the radius is still 11 to 22 cells
wide: the 5 arcmin network of a large country at 200 km has 75 million arcs
and needs 7 GB. Whether animals walk to the residue or the residue is
hauled to them is the same number: R is the distance either may cover.

The map arrays hold, on 0.5-degree cells from 75N to 60S, the residue and
demand (GJ ME per year) and access50: the share of each cell's removable
residue allocated within MAP_RADIUS (50 km) to the pre-catastrophe herd,
each head at the residue it can eat (the demand ratio r / tau of the model
at central parameters, r = mu s as in model/prepare.reach_scale),
pre-catastrophe and per scenario and year.
"""

import numpy as np
import pandas as pd

from src.inputs import crop_tables, crop_tonnes, read_data
from src.model import ratios

# the demand ratios of the tables: 1.024e-4 to 20, 62 points a factor 1.22 apart, 1.0 included
RATIOS = tuple(float(f"{0.05 * 400 ** (k / 30):.4g}") for k in range(-31, 31))
SPLIT = 0.05                              # the tables' ratios below this are solved in a sweep and finer unit of their own
MAP_RADIUS = 50                           # km: the radius of the map arrays (access50) and of Fig. 2
SPECIES = ("cattle", "buffalo", "sheep", "goats")
SPECIES_HERDS = {
    "cattle": ("dairy_cattle", "beef_cattle"), "buffalo": ("dairy_buffaloes", "meat_buffaloes"),
    "sheep": ("dairy_small_ruminants", "meat_small_ruminants"),
    "goats": ("dairy_small_ruminants", "meat_small_ruminants"),
}


def residue_grid(country, spam, per_tonne, outside):
    """Residue ME per pixel (GJ): SPAM production per crop times GJ per fresh tonne,
    per_tonne (C,K) inside the countries and outside (K,) elsewhere."""
    pix = country[spam["row"], spam["col"]]
    inside = pix >= 0
    residue = np.zeros(country.shape, dtype="float32")
    np.add.at(residue, (spam["row"][inside], spam["col"][inside]), (spam["prod"][inside] * per_tonne[pix[inside]]).sum(axis=1))
    np.add.at(residue, (spam["row"][~inside], spam["col"][~inside]), (spam["prod"][~inside] * outside[None, :]).sum(axis=1))
    return residue


def per_tonne(tg, t, fv, crops, iso):
    """(C,K) GJ ME per fresh tonne of SPAM production in scenario tg, year t, and (K,) outside the countries.

    Year t scales each crop by the country's change in the crop tables; a country-crop without
    pre-catastrophe tonnes takes the world change of the crop."""
    order = [crop_tables(47).crops.index(c) for c in crops]
    rpr = pd.DataFrame(ratios.central(tg)[:, order, t], index=iso, columns=crops)
    dmme = (fv["dm_product"] * fv["me"]).to_numpy()
    if t == 0:
        return (rpr * dmme).fillna(0.0).to_numpy(), (fv["dm_product"] * fv["rpr"] * fv["me"]).fillna(0.0).to_numpy()
    tonnes = crop_tonnes(tg)[:, order, :]
    world = tonnes[:, :, t].sum(axis=0) / np.where(tonnes[:, :, 0].sum(axis=0) > 0, tonnes[:, :, 0].sum(axis=0), 1.0)
    change = np.where(tonnes[:, :, 0] > 0, tonnes[:, :, t] / np.where(tonnes[:, :, 0] > 0, tonnes[:, :, 0], 1.0), world[None, :])
    inside = (rpr * dmme).fillna(0.0).to_numpy() * change
    outside = (fv["dm_product"] * fv["rpr"] * fv["me"]).fillna(0.0).to_numpy() * world
    return inside, outside


def aggregate(a, f):
    """Sum f x f blocks of a 2-D array whose sides are multiples of f."""
    a = np.asarray(a, float)
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f).sum(axis=(1, 3))


def country_box(box, shape, f):
    """A slice pair covering box, aligned to multiples of f, within shape."""
    lo0, hi0 = box[0].start // f * f, -(-box[0].stop // f) * f
    lo1, hi1 = box[1].start // f * f, -(-box[1].stop // f) * f
    return slice(lo0, min(hi0, shape[0])), slice(lo1, min(hi1, shape[1]))


def country_grids(country, residue, demand, c, box, f, lat):
    """Supply and demand of country c within box, summed over f x f cells inside its mask, and the
    row latitudes of the result; the whole longitude range when the box spans more than half the globe."""
    if box[1].stop - box[1].start > country.shape[1] // 2:
        box = (box[0], slice(0, country.shape[1]))
    box = country_box(box, country.shape, f)
    mask = country[box] == c
    supply = np.where(mask, residue[box], 0.0)
    need = np.where(mask, demand[box], 0.0)
    lat_rows = lat[box[0]]
    if f > 1:
        supply, need = aggregate(supply, f), aggregate(need, f)
        lat_rows = lat_rows.reshape(-1, f).mean(axis=1)
    return supply, need, lat_rows, box


def demand_grids(country, iso, grid):
    """Pre-catastrophe ruminant ME demand per pixel (GJ per year) by species, and the total."""
    herds = read_data("ruminant_herds.csv")
    by_species = {}
    for sp in SPECIES:
        sub = herds[herds.herd.isin(SPECIES_HERDS[sp])]
        e = (sub["heads"] * sub["me_prewar_mj_d"]).groupby(sub["iso3"]).sum() / sub.groupby("iso3")["heads"].sum().replace(0, np.nan)
        per_head = e.reindex(iso).fillna(e.median()).to_numpy() * 365 / 1000.0                  # GJ per head per year
        heads = np.load(grid / f"heads_{sp}.npy")
        by_species[sp] = np.where(country >= 0, heads * per_head[np.clip(country, 0, None)], 0.0).astype("float32")
    return by_species, sum(by_species.values())
