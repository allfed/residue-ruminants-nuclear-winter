"""Figure 7: where the food is saved. Ten-year mean per person of each country, 47 Tg (Figs. S1 and S2 at 16 and 150 Tg)."""
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from src import inputs as I
from src.figures.maps import MapContext
from src.figures.style import BRACKET, panel_label, plt, save

NAMES = {47: "fig7_map_saved", 16: "figS1_map_saved_16", 150: "figS2_map_saved_150"}


def main(tg=47):
    c = pd.read_csv(I.RESULTS / "core_country.csv")
    c["total"] = c.grain + c.milk + c.meat + c.carcass
    ctx = MapContext()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.3))
    fig.subplots_adjust(hspace=0.08, top=0.96, bottom=0.02)
    bounds = [1, 3, 10, 30, 100, 300]                         # below 1 (the few small losses included) and above 300 extend
    cmap = ListedColormap(["#f7f5ee", "#fde9a6", "#f9c95a", "#eda100", "#d97a00", "#b34d00", "#7a2a00"])
    norm = BoundaryNorm(bounds, cmap.N, extend="both")
    for k, (ax, b) in enumerate(zip(axes, ("kept", "gone"))):
        s = c[(c.scenario == tg) & (c.bracket == b)].groupby("iso3").total.mean()
        ctx.choropleth(ax, s, cmap, norm)
        ax.set_title(f"{tg} Tg, {BRACKET[b]}", pad=4)
        ctx.finish(ax)
        panel_label(ax, "ab"[k], x=0.0, y=1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.02, fraction=0.03, ticks=bounds, extend="both")
    cb.set_ticklabels([f"{b:g}" for b in bounds])
    cb.set_label("food saved, kcal per person of the country per day")
    save(fig, NAMES[tg])


if __name__ == "__main__":
    main()
