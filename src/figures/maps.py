"""Shared map machinery: Winkel Tripel frame, country outlines, choropleths."""
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pyproj import CRS, Transformer

from src import inputs as I

WINKEL_TRIPEL = CRS.from_proj4("+proj=wintri +datum=WGS84 +units=m +no_defs")
MAP_CODE_ALIASES = {"SWT": "SWZ", "XKX": "KOS"}
LAT_BOTTOM, LAT_TOP = -58, 75
LAND = "#f3f2ee"
BORDER = "#9e9d98"


CRIMEA_BOX = (32.0, 44.0, 37.0, 46.5)   # lon/lat box around the peninsula, for picking its polygon


def crimea_in_ukraine(ne):
    """Natural Earth's default (de facto) layer draws Crimea inside Russia; the maps draw it in Ukraine, as in UN
    General Assembly resolution 68/262. Plotting only: the reach grid (reach/grid.py) keeps the layer as it is,
    and Crimea holds about 0.1% of the world's residue there."""
    rus = ne[ne["iso3"] == "RUS"].explode(index_parts=False)
    p = rus.geometry.representative_point()
    x0, y0, x1, y1 = CRIMEA_BOX
    crimea = p.x.between(x0, x1) & p.y.between(y0, y1)
    if not crimea.any():
        return ne
    ukr = ne[ne["iso3"] == "UKR"]
    merged = pd.concat([ukr, rus[crimea].assign(iso3="UKR")]).dissolve(by="iso3", as_index=False)
    rest = rus[~crimea].dissolve(by="iso3", as_index=False)
    others = ne[~ne["iso3"].isin(["RUS", "UKR"])]
    return gpd.GeoDataFrame(pd.concat([others, merged, rest], ignore_index=True), crs=ne.crs)


class MapContext:
    """Country geometry and the projected frame, loaded once per figure."""

    def __init__(self):
        ne = gpd.read_file(I.DATA / "map_context" / "countries.geojson")
        ne["iso3"] = ne["ISO_A3_EH"].where(ne["ISO_A3_EH"] != "-99", ne["ADM0_A3"])
        ne = crimea_in_ukraine(ne)
        # France's outline carries its overseas departments (French Guiana, the Antilles, Reunion, Mayotte);
        # the model's France is one country, so they are drawn apart and keep the land colour
        fra = ne[ne["iso3"] == "FRA"].explode(index_parts=False)
        c = fra.geometry.centroid
        europe = c.x.between(-10, 20) & c.y.between(35, 55)
        overseas = fra[~europe].assign(iso3="FRA_overseas")
        fra = fra[europe].dissolve(by="iso3", as_index=False)
        ne = gpd.GeoDataFrame(pd.concat([ne[ne["iso3"] != "FRA"], fra, overseas], ignore_index=True), crs=ne.crs)
        self.countries = ne[ne["iso3"] != "ATA"].to_crs(WINKEL_TRIPEL)
        self.transform = Transformer.from_crs("EPSG:4326", WINKEL_TRIPEL, always_xy=True)
        self.frame = self._frame()
        v = self.frame.vertices
        self.bounds = (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())

    def _frame(self):
        lon = np.linspace(-180, 180, 721)
        lat = np.linspace(LAT_BOTTOM, LAT_TOP, 361)
        t = self.transform.transform
        segments = (t(lon, np.full_like(lon, LAT_BOTTOM)), t(np.full_like(lat, 180), lat),
                    t(lon[::-1], np.full_like(lon, LAT_TOP)), t(np.full_like(lat, -180), lat[::-1]))
        vertices = np.column_stack((np.concatenate([s[0] for s in segments]), np.concatenate([s[1] for s in segments])))
        return Path(vertices, closed=True)

    def grid(self, shape, top):
        """Projected coordinates of the edges of 0.5-degree cells."""
        nlat, nlon = shape
        lon = np.linspace(-180, 180, nlon + 1)
        lat = top - np.arange(nlat + 1) * 0.5
        lon2d, lat2d = np.meshgrid(lon, lat)
        return self.transform.transform(lon2d, lat2d)

    def land(self, ax, edgecolor="none"):
        self.countries.plot(ax=ax, color=LAND, edgecolor=edgecolor, linewidth=0.22, zorder=0)

    def borders(self, ax):
        self.countries.boundary.plot(ax=ax, color=BORDER, linewidth=0.22, zorder=2)

    def choropleth(self, ax, values, cmap, norm, missing="white"):
        """Fill countries by a Series indexed by iso3; countries without a value (not modelled, e.g. Greenland)
        are left white within their borders, so they do not read as the lightest class."""
        s = values.rename(index=MAP_CODE_ALIASES)
        g = self.countries.merge(s.rename("value"), left_on="iso3", right_index=True, how="left")
        self.countries.plot(ax=ax, color=missing, edgecolor=BORDER, linewidth=0.22, zorder=0)
        has = g.dropna(subset=["value"])
        has.plot(ax=ax, column="value", cmap=cmap, norm=norm, edgecolor=BORDER, linewidth=0.22, zorder=1)
        return g

    def finish(self, ax):
        clip = PathPatch(self.frame, transform=ax.transData, facecolor="none", edgecolor="none")
        ax.add_patch(clip)
        for collection in ax.collections:
            collection.set_clip_path(clip)
        t = self.transform.transform
        lon = np.linspace(-180, 180, 721)
        for latitude in (-30, 0, 30, 60):
            ax.plot(*t(lon, np.full_like(lon, latitude)), color="#dfded9", linewidth=0.35, zorder=-1)
        lat = np.linspace(LAT_BOTTOM, LAT_TOP, 361)
        for longitude in (-120, -60, 0, 60, 120):
            ax.plot(*t(np.full_like(lat, longitude), lat), color="#dfded9", linewidth=0.35, zorder=-1)
        ax.plot(self.frame.vertices[:, 0], self.frame.vertices[:, 1], color="#aaa9a4", linewidth=0.45, zorder=5)
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
