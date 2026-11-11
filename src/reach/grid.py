"""Common 5 arcmin grid (2160 x 4320, the GLW4 grid) of the reach allocation (access.py) and Fig. 2.

Writes to work/grid/: country index raster (Natural Earth 10m admin 0,
rasterised, index into inputs.countries(), -1 elsewhere), pixel area,
ruminant heads per species (GLW4 2020 dasymetric densities times pixel
area) and SPAM 2020 production per crop (yield times harvested area, all
technologies) put on the same grid.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_origin

from src.inputs import WORK, countries, crop_tables

NE = os.environ.get("RR_NATURAL_EARTH", str(WORK / "raw/natural-earth/ne_10m_admin_0_countries.shp"))
SPAM = os.environ.get("RR_SPAM", str(WORK / "raw/SPAM"))
SHAPE = (2160, 4320)
RES = 1 / 12
GLW = {"cattle": "CTL", "buffalo": "BFL", "sheep": "SHP", "goats": "GTS"}
GRID = WORK / "grid"


def pixel_area_km2():
    lat_edges = np.deg2rad(90 - np.arange(SHAPE[0] + 1) * RES)
    band = 6371.0088 ** 2 * np.deg2rad(RES) * (np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:]))
    return np.repeat(band[:, None], SHAPE[1], axis=1).astype("float32")


def country_raster():
    ne = gpd.read_file(NE)
    iso = ne["ISO_A3_EH"].where(ne["ISO_A3_EH"] != "-99", ne["ADM0_A3"])
    index = {c: i for i, c in enumerate(countries())}
    shapes = [(g, index[c]) for g, c in zip(ne.geometry, iso) if c in index]
    tr = from_origin(-180, 90, RES, RES)
    return features.rasterize(shapes, out_shape=SHAPE, transform=tr, fill=-1, dtype="int16", all_touched=False)


def glw_heads(area):
    out = {}
    for sp, code in GLW.items():
        with rasterio.open(WORK / "raw" / f"GLW4-2020.D-DA.{code}.tif") as r:
            a = r.read(1)
        a = np.where((a < 0) | ~np.isfinite(a), 0, a)
        out[sp] = (a * area).astype("float32")
    return out


def spam_production():
    crops = crop_tables(47).crops
    ycols = [f"{c}_A" for c in crops]
    use = ["grid_code", "x", "y"] + ycols
    y = pd.read_csv(f"{SPAM}/spam2020V1r0_global_Y_TA.csv", usecols=use, encoding="utf-8-sig")
    h = pd.read_csv(f"{SPAM}/spam2020V1r0_global_harvested_area/spam2020V1r0_global_H_TA.csv",
                    usecols=use, encoding="utf-8-sig")
    h = h.set_index("grid_code")[ycols].reindex(y["grid_code"]).fillna(0.0)
    prod = y[ycols].to_numpy("float32") * h.to_numpy("float32") / 1000.0   # tonnes
    row = np.floor((90 - y.y.to_numpy()) / RES).astype(int)
    col = np.floor((y.x.to_numpy() + 180) / RES).astype(int)
    return crops, row, col, prod


def main():
    GRID.mkdir(parents=True, exist_ok=True)
    area = pixel_area_km2()
    np.save(GRID / "area.npy", area)
    cr = country_raster()
    np.save(GRID / "country.npy", cr)
    print("country pixels", (cr >= 0).sum(), "countries on raster", len(np.unique(cr)) - 1)
    if not (GRID / "heads_goats.npy").exists():
        for sp, a in glw_heads(area).items():
            np.save(GRID / f"heads_{sp}.npy", a)
            print(sp, a.sum() / 1e6, "M head; inside countries", a[cr >= 0].sum() / 1e6)
    crops, row, col, prod = spam_production()
    np.savez_compressed(GRID / "spam_prod.npz", crops=np.array(crops), row=row, col=col, prod=prod)
    print("SPAM Mt:", dict(zip(crops[:6], (prod.sum(0)[:6] / 1e6).round(0))))


if __name__ == "__main__":
    main()
