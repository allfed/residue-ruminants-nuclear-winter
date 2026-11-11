"""Figure 2: the reach allocation at native resolution, year 4 after 47 Tg, grain kept.

The allocation the model applies, cell by cell, in the central run, in the grid's own terms:
(a) demand: the residue ME the surviving animals of each cell could eat, mu s d_j, with d_j the
gridded pre-catastrophe demand, mu the country's ratio in that year of the central run
(results/core_country.csv, reach_mu_iv) and s = E0 E / (S D) (model.prepare.reach_scale), which
puts what the animals can eat nationally, Cap = mu E0, on the gridded residue's level: mu s d_j
is Cap d_j / D times E / S,
(b) supply: the residue ME of each cell that may leave the field, tau e_i, with e_i the gridded
residue of the year and tau one minus the shares left on the field and taken by other uses
(model.prepare at central parameters),
(c) the share of each cell's supply tau e_i which the maximum flow assigns to animals of the same
country within 50 km: cell by cell, the model's g(mu s / tau).
One 50 km reach circle around an animal cell is drawn in every panel and labelled in (a).

The flow is solved exactly for each country whole, tau e_i against mu s d_j, with the functions of
reach/access.py; it is homogeneous in its capacities, so the national levels (S and Cap in place of
E and mu s D) scale every cell by S / E and leave the shares of (c) unchanged. The model interpolates
g between tabulated ratios, which can only understate it (by at most the interpolation bound of
residue_access.csv). The total of a
maximum flow is unique but its split between cells is not: where supply exceeds what the
animals in reach can eat, (c) is one of the allocations that feed the most. Transfers stay
within a country, so the borders are drawn in every panel.

Needs the 5 arcmin grid of reach/grid.py (work/grid: SPAM 2020 production,
GLW4 2020 heads, the country raster) and the central run's results/core_country.csv.

Any other region, scenario, year and grain case (the defaults draw the figure of the paper):

    python -m src.figures.fig2_reach_inputs --window LON_MIN LON_MAX LAT_MIN LAT_MAX \
        --circle-at LON LAT --scenario 16|47|150 --year 1-10 --bracket kept|gone --name NAME
"""
from __future__ import annotations

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LogNorm, Normalize
from pyproj import CRS, Geod, Transformer
from scipy.ndimage import find_objects

from src import inputs as I
from src.reach import access as A
from src.reach.flow import ReachNetwork
from src.figures.maps import LAND
from src.figures.style import plt, save

SCENARIO, YEAR, BRACKET = 47, 4, "kept"
RADIUS_KM = float(A.MAP_RADIUS)
GRID = I.WORK / "grid"
WINDOW = (-10.0, 4.0, 35.5, 44.5)       # lon_min, lon_max, lat_min, lat_max: Iberia, southern France, northern Morocco
CIRCLE_AT = (-4.29, 39.625)            # lon, lat: the reach is drawn around the animal cell nearest this point, a dense demand spot
CELL_NORM = LogNorm(2, 150)              # TJ ME per 5 arcmin cell per year, (a) and (b)
CELL_CMAP = "YlGnBu"
SHARE_NORM = Normalize(0, 1)
BORDER_DARK = "#52514e"


def window(box, margin=(0.0, 0.0)):
    """Row and column slices of the 5 arcmin grid covering box widened by margin (degrees lon, lat)."""
    x0, x1, y0, y1 = box
    lat = 90 - (np.arange(2160) + 0.5) / 12
    lon = -180 + (np.arange(4320) + 0.5) / 12
    r = np.flatnonzero((lat >= y0 - margin[1]) & (lat <= y1 + margin[1]))
    c = np.flatnonzero((lon >= x0 - margin[0]) & (lon <= x1 + margin[0]))
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def allocation(box=WINDOW):
    """Supply tau e_i, demand mu s d_j and allocated residue (GJ ME per cell per year, the grid's own terms) on
    the 5 arcmin grid for the countries with a cell in box, and the country raster."""
    import warnings
    import pandas as pd
    from src.model.mechanism import access_share
    from src.model.prepare import prepare
    iso = I.countries()
    n = len(iso)
    country = np.load(GRID / "country.npy")
    spam = dict(np.load(GRID / "spam_prod.npz"))
    crops = list(spam["crops"])
    fv = I.read_data("residue_feed_values.csv").set_index("crop_code").reindex(crops)
    _, demand = A.demand_grids(country, iso, GRID)
    residue = A.residue_grid(country, spam, *A.per_tonne(SCENARIO, YEAR, fv, crops, iso)).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = prepare(SCENARIO)
    if list(p.iso3) != iso:
        raise ValueError("The model's countries differ from the grid's")
    tau = np.clip(np.asarray(p.tau, float), 0.0, 1.0)
    core = pd.read_csv(I.RESULTS / "core_country.csv")
    core = core[(core.scenario == SCENARIO) & (core.year == YEAR) & (core.bracket == BRACKET)].set_index("iso3")
    mu = core["reach_mu_iv"].reindex(iso).to_numpy(float)
    if p.reach_scale is None:
        raise ValueError("residue_access.csv has no gridded totals (columns residue_pj and demand_pj)")
    r = mu * p.reach_scale[:, YEAR]                                      # the model's table ratio, mu s
    model = access_share(p, YEAR, np.nan_to_num(mu))                     # tau g(mu s / tau), interpolated
    ok = country >= 0
    lat = 90 - (np.arange(country.shape[0]) + 0.5) / 12
    boxes = find_objects(country.astype(int) + 1, max_label=n)
    present = np.unique(country[window(box)])
    allocated = np.zeros_like(residue)
    for k in present[present >= 0]:
        supply, need, lat_rows, b = A.country_grids(country, residue, demand, k, boxes[k], 1, lat)
        if supply.sum() == 0 or need.sum() == 0 or not np.isfinite(r[k]) or tau[k] <= 0 or r[k] <= 0:
            continue
        net = ReachNetwork(supply, need, RADIUS_KM, lat_rows, wrap=b[1].stop - b[1].start == country.shape[1])
        total, alloc = net.solve(r=float(r[k]), tau=float(tau[k]))         # tau e_i against mu s d_j
        cell = np.zeros(supply.size)
        cell[net.src] = alloc
        allocated[b] += cell.reshape(supply.shape)
        print(f"{iso[k]}: tau {tau[k]:.2f}, mu {mu[k]:.3f}, s {p.reach_scale[k, YEAR]:.3f}, available residue allocated within "
              f"{RADIUS_KM:.0f} km {total / (tau[k] * supply.sum()):.1%} (model, interpolated: {model[k] / tau[k]:.1%})")
    cty = np.clip(country, 0, None)
    available = np.where(ok, residue * tau[cty], 0.0)                                 # tau e_i
    eat = np.where(ok, demand * np.nan_to_num(r)[cty], 0.0)                           # mu s d_j
    return available, eat, allocated, country


def regional_crs(box):
    x0, x1, y0, y1 = box
    return CRS.from_proj4(f"+proj=laea +lat_0={(y0 + y1) / 2} +lon_0={(x0 + x1) / 2} +datum=WGS84 +units=m +no_defs")


def main(name="fig2_reach_inputs"):
    residue, eat, allocated, country = allocation(WINDOW)
    box = WINDOW
    x0, x1, y0, y1 = box
    iso = np.array(I.countries())
    rw, cw = window(box)
    inside = country[rw, cw] >= 0
    for k in np.unique(country[rw, cw][inside]):
        m = country[rw, cw] == k
        if residue[rw, cw][m].sum() < 1e5:   # under 0.1 PJ: a few border cells
            continue
        print(f"window, {iso[k]}: available residue {residue[rw, cw][m].sum() / 1e6:.1f} PJ, the animals could eat {eat[rw, cw][m].sum() / 1e6:.1f} PJ, "
              f"allocated {allocated[rw, cw][m].sum() / residue[rw, cw][m].sum():.0%} of the available residue")
    crs = regional_crs(box)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    r, c = window(box, (1.0, 0.5))
    lat_e = 90 - np.arange(r.start, r.stop + 1) / 12
    lon_e = -180 + np.arange(c.start, c.stop + 1) / 12
    X, Y = tr.transform(*np.meshgrid(lon_e, lat_e))
    xlim = (tr.transform(x0, (y0 + y1) / 2)[0], tr.transform(x1, (y0 + y1) / 2)[0])
    ylim = (tr.transform((x0 + x1) / 2, y0)[1], tr.transform((x0 + x1) / 2, y1)[1])
    import geopandas as gpd
    countries = gpd.read_file(I.DATA / "map_context" / "countries.geojson").clip_by_rect(x0 - 6, y0 - 4, x1 + 6, y1 + 4)
    countries = gpd.GeoSeries(countries[~countries.is_empty], crs="EPSG:4326").to_crs(crs)

    # layout in inches
    W, pad, gap = 7.5, 0.05, 0.1
    w = (W - 2 * pad - 2 * gap) / 3
    h = w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    bottom_h, top_h = 0.78, 0.24
    H = bottom_h + h + top_h
    fig = plt.figure(figsize=(W, H))

    def add(x, y, wx, hy):
        return fig.add_axes((x / W, y / H, wx / W, hy / H))
    axes = [add(pad + k * (w + gap), bottom_h, w, h) for k in range(3)]
    tj = 1e-3   # GJ to TJ
    share = np.where(residue[r, c] > 0, allocated[r, c] / np.maximum(residue[r, c], 1e-12), np.nan)
    panels = ((eat[r, c] * tj, CELL_NORM, CELL_CMAP, "Demand: what the animals can eat"),
              (residue[r, c] * tj, CELL_NORM, CELL_CMAP, "Supply: residue available"),
              (np.clip(share, 0, 1), SHARE_NORM, "viridis", "Share of residue allocated to animals"))
    ims = []
    for k, (ax, (a, norm, cmap, title)) in enumerate(zip(axes, panels)):
        countries.plot(ax=ax, color=LAND, edgecolor="none", zorder=0)
        shown = np.where(a > 0, a, np.nan) if k < 2 else a
        ims.append(ax.pcolormesh(X, Y, shown, norm=norm, cmap=cmap, shading="flat", edgecolor="none", rasterized=True, zorder=1))
        countries.boundary.plot(ax=ax, color="white", linewidth=1.5, alpha=0.8, zorder=2)
        countries.boundary.plot(ax=ax, color=BORDER_DARK, linewidth=0.7, zorder=3)
        ax.set_title(f"$\\bf{{{'abc'[k]}}}$  {title}", loc="left", pad=3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#aaa9a4")
            spine.set_linewidth(0.45)

    # one 50 km reach around the animal cell nearest CIRCLE_AT
    lat_c = 90 - (np.arange(r.start, r.stop) + 0.5) / 12
    lon_c = -180 + (np.arange(c.start, c.stop) + 0.5) / 12
    LON, LAT = np.meshgrid(lon_c, lat_c)
    dist = np.hypot((LON - CIRCLE_AT[0]) * np.cos(np.deg2rad(CIRCLE_AT[1])), LAT - CIRCLE_AT[1])
    i, j = np.unravel_index(np.argmin(np.where(eat[r, c] > 0, dist, np.inf)), dist.shape)
    az = np.linspace(0, 360, 181)
    lon_r, lat_r, _ = Geod(ellps="WGS84").fwd(np.full_like(az, lon_c[j]), np.full_like(az, lat_c[i]), az, np.full_like(az, RADIUS_KM * 1000))
    print(f"reach circle around the cell centred on {lat_c[i]:.3f}, {lon_c[j]:.3f}")
    for ax in axes:
        ax.plot(*tr.transform(lon_r, lat_r), color="white", linewidth=2.0, zorder=4)
        ax.plot(*tr.transform(lon_r, lat_r), color="#0b0b0b", linewidth=0.9, zorder=5)
    axes[0].plot(*tr.transform(lon_c[j], lat_c[i]), marker="+", color="#0b0b0b", markersize=5, markeredgewidth=0.9, zorder=5,
                 path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])
    # label on a, just outside the circle to the upper right
    lx, ly = tr.transform(*Geod(ellps="WGS84").fwd(lon_c[j], lat_c[i], 45, RADIUS_KM * 1000 * 1.15)[:2])
    axes[0].text(lx, ly, f"{RADIUS_KM:.0f} km reach", fontsize=7.5, color="#0b0b0b", ha="left", va="bottom", zorder=6,
                 path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])

    # one colourbar under each panel; a and b on the same scale
    cb_y, cb_h = bottom_h - 0.2, 0.08
    labels = ("demand for residue,\nTJ ME per cell per year", "residue that may leave the field,\nTJ ME per cell per year",
              "share of the cell's available\nresidue allocated")
    for k, (im, label) in enumerate(zip(ims, labels)):
        cb = fig.colorbar(im, cax=add(pad + k * (w + gap) + 0.1 * w, cb_y, 0.8 * w, cb_h), orientation="horizontal",
                          extend="both" if k < 2 else "neither")
        if k < 2:
            cb.set_ticks([v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if CELL_NORM.vmin <= v <= CELL_NORM.vmax])
            cb.ax.xaxis.set_minor_formatter(plt.NullFormatter())
        cb.ax.xaxis.set_major_formatter(plt.FuncFormatter((lambda v, _: f"{v:g}") if k < 2 else (lambda v, _: f"{v:.0%}")))
        cb.set_label(label, labelpad=2)

    save(fig, name)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Figure 2 for any window, scenario, year and grain case; "
                                             "the defaults draw the figure of the paper.")
    ap.add_argument("--window", type=float, nargs=4, default=WINDOW, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    ap.add_argument("--circle-at", type=float, nargs=2, default=CIRCLE_AT, metavar=("LON", "LAT"),
                    help="the reach circle is drawn around the animal cell nearest this point")
    ap.add_argument("--scenario", type=int, choices=I.SCENARIOS, default=SCENARIO, help="Tg of soot")
    ap.add_argument("--year", type=int, choices=range(1, I.N_YEARS + 1), default=YEAR, metavar="1-10")
    ap.add_argument("--bracket", choices=("kept", "gone"), default=BRACKET, help="grain kept or grain gone")
    ap.add_argument("--name", default="fig2_reach_inputs", help="file name in figures/, without extension")
    a = ap.parse_args()
    WINDOW, CIRCLE_AT = tuple(a.window), tuple(a.circle_at)
    SCENARIO, YEAR, BRACKET = a.scenario, a.year, a.bracket
    main(a.name)
