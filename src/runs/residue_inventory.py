"""Pre-catastrophe world residue inventory, per crop (the paper's Table 1 and Table S1).

Product (fresh Mt, baseline year of the crop tables) times the dry-matter
fraction of the product times the residue ratio at each country's pre-catastrophe
yield (model/ratios.py; central, low, high), summed over countries, with the
feed values of the residue. The world ratio is the resulting residue over
product dry matter. Writes results/residue_inventory.csv.

Validation: the cereal residue of this inventory, per country, against the
2019-2021 cereal residue of Smerald et al. (2023) on the same 169 countries
(data/benchmark_smerald.csv), for the world and eight regions (GROUPS).
Writes results/residue_benchmark.csv.
"""
import numpy as np
import pandas as pd

from src import inputs as I
from src.model import ratios

CEREALS = ["MAIZ", "RICE", "WHEA", "BARL", "SORG", "OCER", "PMIL", "MILL"]
# eight world regions: the GLEAM regions, with Eastern Europe and Russia, and North America and Oceania, merged
GROUPS = {"SA": "South Asia", "SSA": "Sub-Saharan Africa", "NENA": "Near East and North Africa",
          "ESEA": "East and Southeast Asia", "LAC": "Latin America", "WE": "Western Europe",
          "EE": "Eastern Europe and Russia", "RUSS": "Eastern Europe and Russia",
          "NA": "North America and Oceania", "OC": "North America and Oceania"}


def country_residue(fv, ct):
    """(C,K) pre-catastrophe residue, Mt DM, at the central, low and high ratios."""
    dm = I.crop_tonnes(47)[:, :, 0] / 1e6 * fv.reindex(ct.crops)["dm_product"].fillna(0.0).to_numpy()[None, :]
    return {k: dm * ratios.positioned(47, pos)[:, :, 0] for k, pos in (("ours", 0.0), ("ours_low", -1.0), ("ours_high", 1.0))}


def benchmark(fv, ct):
    """Cereal residue per region and world: this inventory (central, low, high) and Smerald et al. (2023)."""
    col = [list(ct.crops).index(c) for c in CEREALS]
    res = country_residue(fv, ct)
    ours = pd.DataFrame({k: res[k][:, col].sum(axis=1) for k in ("ours", "ours_low", "ours_high")}, index=I.countries())
    sm = I.read_data("benchmark_smerald.csv").set_index("iso3")
    methods = ["fixed_ratio", "exponential", "linear"]
    df = ours.join(sm[["mean"] + methods].rename(columns={"mean": "smerald"}))
    df["region"] = [GROUPS[I.regions()[c]] for c in df.index]
    out = pd.concat([df.groupby("region").sum(numeric_only=True).reindex(list(dict.fromkeys(GROUPS.values()))),
                     df.sum(numeric_only=True).rename("World").to_frame().T])
    out["smerald_low"], out["smerald_high"] = out[methods].min(axis=1), out[methods].max(axis=1)
    out["ours_over_smerald"] = out.ours / out.smerald
    out = out[["ours", "ours_low", "ours_high", "smerald", "smerald_low", "smerald_high", "ours_over_smerald"]]
    out.loc["Outside the 169 countries (Smerald et al. only)"] = [np.nan] * 3 + [sm.loc["outside", "mean"],
                                                                     sm.loc["outside", methods].min(), sm.loc["outside", methods].max(), np.nan]
    return out.rename_axis("region")


def main():
    fv = I.read_data("residue_feed_values.csv").set_index("crop_code")
    ct = I.crop_tables(47)
    prod = pd.Series(I.crop_tonnes(47)[:, :, 0].sum(axis=0) / 1e6, index=list(ct.crops))
    prod = prod.reindex(fv.index)
    res = {k: pd.Series(v.sum(axis=0), index=list(ct.crops)).reindex(fv.index) for k, v in country_residue(fv, ct).items()}
    rpr = (res["ours"] / (prod * fv.dm_product)).where(prod * fv.dm_product > 0, 0.0)
    out = pd.DataFrame({"residue": fv.residue, "prod_Mt": prod, "dm_low": res["ours_low"], "dm": res["ours"],
                        "dm_high": res["ours_high"], "rpr": rpr, "rpr_ref": fv.rpr, "me": fv.me, "cp": fv.cp, "ndf": fv.ndf,
                        "cls": fv.feed_class})
    out = out.sort_values("dm", ascending=False).round(1)
    I.RESULTS.mkdir(exist_ok=True)
    out.to_csv(I.RESULTS / "residue_inventory.csv")
    used = out[out.dm > 0]
    print(f"{len(used)} crops with a residue, {used.dm.sum():.0f} Mt DM ({used.dm_low.sum():.0f} to {used.dm_high.sum():.0f})")
    b = benchmark(fv, ct)
    b.round(3).to_csv(I.RESULTS / "residue_benchmark.csv")
    print("cereal residue, Mt DM: this inventory against Smerald et al. 2023 (2019-2021)")
    print(b.round(2).to_string())


if __name__ == "__main__":
    main()
