"""Paths and readers of the inputs in data/.

The upstream drivers (crop and pasture production per scenario, pre-catastrophe
livestock products, the countries and their regions) are in ``data/drivers``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
WORK = ROOT / "work"
DRIVERS = DATA / "drivers"

SCENARIOS = (16, 47, 150)
N_YEARS = 10
HERDS = (
    "dairy_cattle", "beef_cattle", "dairy_buffaloes", "meat_buffaloes",
    "dairy_small_ruminants", "meat_small_ruminants",
)
#: Product groups per herd in the FAOSTAT baseline.
MILK_OF = {
    "dairy_cattle": ("cow_milk",),
    "dairy_buffaloes": ("buffalo_milk",),
    "dairy_small_ruminants": ("sheep_milk", "goat_milk"),
}
MEAT_OF = {
    "beef_cattle": ("cattle_meat",),
    "meat_buffaloes": ("buffalo_meat",),
    "meat_small_ruminants": ("sheep_meat", "goat_meat"),
}
KCAL_PER_TCAL = 1e12
MJ_PER_KCAL = 4.184e-3


@dataclass(frozen=True)
class CropTables:
    iso3: List[str]
    crops: List[str]
    #: fresh tonnes, (country, crop, year) with year 0 the pre-catastrophe baseline
    tonnes: np.ndarray
    #: gross Tcal, same shape
    tcal: np.ndarray


@lru_cache(maxsize=None)
def crop_tables(tg: int) -> CropTables:
    if tg not in SCENARIOS:
        raise ValueError(f"Unknown crop scenario: {tg}")
    with np.load(DRIVERS / f"crops_{tg}Tg.npz", allow_pickle=False) as z:
        return CropTables(z["iso3"].tolist(), z["crops"].tolist(), z["tonnes"], z["tcal"])


@lru_cache(maxsize=None)
def countries() -> List[str]:
    """The 169 countries with crop-production and regional feed-ration data."""
    return pd.read_csv(DRIVERS/"countries.csv", keep_default_na=False).iso3.tolist()


@lru_cache(maxsize=None)
def regions() -> Dict[str, str]:
    df = pd.read_csv(DRIVERS/"countries.csv", keep_default_na=False)
    return dict(zip(df.iso3, df.region))


@lru_cache(maxsize=None)
def grass_ratio(tg: int) -> np.ndarray:
    """(country, year 1..10) grass production over the control run."""
    df = pd.read_csv(DRIVERS/f"grass_{tg}Tg.csv", float_precision="round_trip").set_index("iso3")
    return df.reindex(countries()).to_numpy(float)


@lru_cache(maxsize=None)
def crop_tonnes(tg: int) -> np.ndarray:
    """(country, crop, year 0..10) fresh tonnes on the common country list."""
    ct = crop_tables(tg)
    idx = [ct.iso3.index(c) for c in countries()]
    return ct.tonnes[idx]


@lru_cache(maxsize=None)
def crop_tcal(tg: int) -> np.ndarray:
    """(country, crop, year 0..10) gross Tcal on the common country list."""
    ct = crop_tables(tg)
    idx = [ct.iso3.index(c) for c in countries()]
    return ct.tcal[idx]


@lru_cache(maxsize=None)
def animal_baseline() -> pd.DataFrame:
    """Pre-catastrophe ruminant products per country: tonnes and dietary Tcal."""
    return pd.read_csv(DRIVERS/"animal_baseline.csv", keep_default_na=False, float_precision="round_trip")


#: Most meat a stocked herd yields, carcasses per head per year, by species (herd_products). Model assumption; sheep
#: and goats 2 for twins and young born and slaughtered within the year, which the stock count misses.
MEAT_CEILING = {"cattle": 1.0, "buffaloes": 1.0, "small_ruminants": 2.0}
HERD_SPECIES = {"dairy_cattle": "cattle", "beef_cattle": "cattle", "dairy_buffaloes": "buffaloes", "meat_buffaloes": "buffaloes",
                "dairy_small_ruminants": "small_ruminants", "meat_small_ruminants": "small_ruminants"}
SPECIES_HERDS = (("beef_cattle", "dairy_cattle"), ("meat_buffaloes", "dairy_buffaloes"),
                 ("meat_small_ruminants", "dairy_small_ruminants"))
CAPPED_COLUMNS = ["iso3", "herd", "heads", "carcasses_per_head", "ceiling", "meat_t", "meat_tcal", "ceiling_tcal",
                  "to_herd", "moved_t", "moved_tcal", "dropped_t", "dropped_tcal"]


def herd_products(herds: pd.DataFrame, report: bool = False):
    """animal_baseline() as the model reads it, given the herd table (iso3, herd, heads, lw_kg, dressing,
    carcass_kcal_per_kg). FAOSTAT records meat by species; ruminant_herds.csv splits each species into a dairy and a meat herd.
    1. The meat of an empty meat herd goes to the dairy herd of its species, as the meat of its culls and surplus
       young (ruminant_herds.csv gives the whole stock to the dairy herd where milked animals and followers fill it).
    2. No stocked herd yields more than its species' MEAT_CEILING carcasses per head per year, a carcass being
       lw_kg x dressing x carcass_kcal_per_kg as prepare's carcass_kcal. Meat above it moves to the dairy herd of
       the species, up to the same ceiling there (its meat plus what it receives); the rest is slaughter of imported
       live animals, which would stop, and is dropped. All meat rows of a herd scale by one share, so tonnes follow Tcal.
    Milk is untouched; empty herds keep their products. With report=True, also returns
    one row per capped herd (CAPPED_COLUMNS, tonnes and Tcal per year)."""
    base = animal_baseline().copy()
    base["tonnes"] = base["tonnes"].astype(float)
    h = herds.set_index(["iso3", "herd"])
    n = h["heads"]
    for meat, dairy in SPECIES_HERDS:
        rows = base[base.herd == meat]
        empty = n.reindex(list(zip(rows.iso3, rows.herd))).fillna(0).to_numpy() == 0
        stocked = n.reindex([(c, dairy) for c in rows.iso3]).fillna(0).to_numpy() > 0
        base.loc[rows.index[empty & stocked], "herd"] = dairy
    carcass = h["lw_kg"] * h["dressing"] * h["carcass_kcal_per_kg"].fillna(2000.0)          # kcal per carcass
    ceiling = h.index.get_level_values("herd").map(HERD_SPECIES).map(MEAT_CEILING).to_numpy(float)
    cap = n * carcass * ceiling / KCAL_PER_TCAL                                             # Tcal per year
    is_meat = ~base["product"].isin([p for ps in MILK_OF.values() for p in ps])
    held = base[is_meat].groupby(["iso3", "herd"])["tcal"].sum()
    over = held[(held > cap.reindex(held.index).fillna(0)) & (n.reindex(held.index).fillna(0) > 0)]
    moved_rows, recs = [], []
    for (iso, herd), m in over.items():
        sel = is_meat & (base.iso3 == iso) & (base.herd == herd)
        excess, tonnes = m - cap[(iso, herd)], base.loc[sel, "tonnes"].sum()
        dairy = dict(SPECIES_HERDS).get(herd, "")
        room = max(cap.get((iso, dairy), 0.0) - held.get((iso, dairy), 0.0), 0.0) if dairy else 0.0
        moved = min(excess, room)
        if moved > 0:
            part = base.loc[sel].copy()
            part[["tonnes", "tcal"]] *= moved / m
            part["herd"] = dairy
            moved_rows.append(part)
        base.loc[sel, ["tonnes", "tcal"]] *= 1 - excess / m
        c = MEAT_CEILING[HERD_SPECIES[herd]]
        recs.append(dict(iso3=iso, herd=herd, heads=n[(iso, herd)], carcasses_per_head=c * m / cap[(iso, herd)], ceiling=c,
                         meat_t=tonnes, meat_tcal=m, ceiling_tcal=cap[(iso, herd)], to_herd=dairy if moved > 0 else "",
                         moved_t=tonnes * moved / m, moved_tcal=moved, dropped_t=tonnes * (excess - moved) / m,
                         dropped_tcal=excess - moved))
    if moved_rows:
        base = (pd.concat([base] + moved_rows).groupby(["iso3", "herd", "product"], as_index=False, sort=False)
                [["tonnes", "tcal"]].sum())
    return (base, pd.DataFrame(recs, columns=CAPPED_COLUMNS)) if report else base


@lru_cache(maxsize=None)
def population() -> pd.Series:
    path = DATA / "population.csv"
    df = pd.read_csv(path, comment="#")
    col = [c for c in df.columns if c.lower().startswith("pop")][0]
    values = df.set_index("iso3")[col].reindex(countries())
    if values.isna().any() or (values <= 0).any():
        raise ValueError("Population must cover every modeled country with a positive value")
    return values


def read_data(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, comment="#", keep_default_na=False, na_values=[""])


@lru_cache(maxsize=None)
def counted_crops() -> tuple:
    """Crop codes whose field residue is fed, in the order of the crop tables:
    a feed class and a nonzero residue ratio in residue_feed_values.csv."""
    fv = read_data("residue_feed_values.csv").set_index("crop_code")
    fv = fv.reindex(crop_tables(47).crops)
    return tuple(c for c in fv.index if isinstance(fv.at[c, "feed_class"], str) and fv.at[c, "feed_class"] and fv.at[c, "rpr"] > 0)
