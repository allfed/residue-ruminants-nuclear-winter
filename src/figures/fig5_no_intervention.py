"""Figure 5: what the shock does to the herds when nothing is done."""
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch

from src import inputs as I
from src.figures.maps import MapContext
from src.figures.style import SCENARIO_COLORS, panel_label, plt, save


def main():
    w = pd.read_csv(I.RESULTS / "core_world.csv")
    w["animal_ni"] = w.milk_ni + w.meat_ni
    prewar = w[(w.scenario == 0) & (w.bracket == "kept") & (w.year == 1)].animal_ni.iloc[0]
    fig = plt.figure(figsize=(7.5, 5.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.55], hspace=0.8, wspace=0.28)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_m = fig.add_subplot(gs[1, :])
    for tg in I.SCENARIOS:
        for b, ls in (("kept", "-"), ("gone", "--")):
            s = w[(w.scenario == tg) & (w.bracket == b)]
            ax_h.plot(s.year, 100 * s.herd_ni, color=SCENARIO_COLORS[tg], ls=ls, label=f"{tg} Tg, grain {b}")
            ax_a.plot(s.year, s.animal_ni, color=SCENARIO_COLORS[tg], ls=ls)
            if b == "kept":
                ax_a.fill_between(s.year, s.animal_ni, s.animal_ni + s.carcass_ni, color=SCENARIO_COLORS[tg], alpha=0.3, lw=0)
    ax_a.axhline(prewar, color="#8a8984", lw=0.8, ls=":")
    ax_a.text(10, prewar + 4, "pre-catastrophe", va="bottom", ha="right", fontsize=7, color="#52514e")
    ax_h.set_ylim(0, 100); ax_a.set_ylim(0, 1.12 * prewar)
    ax_h.set_ylabel("ruminants alive,\n% of pre-catastrophe")
    ax_a.set_ylabel("milk and meat,\nkcal per person per day")
    for ax in (ax_h, ax_a):
        ax.set_xlabel("year after the war")
        ax.set_xticks(range(1, 11))
    h, l = ax_h.get_legend_handles_labels()
    h.append(Patch(color="#8a8984", alpha=0.35, lw=0)); l.append("meat from culled animals,\ngrain kept (panel b)")
    fig.legend(h, l, loc="center", bbox_to_anchor=(0.5, 0.548), ncol=4, fontsize=7, handlelength=2.4)
    panel_label(ax_h, "a"); panel_label(ax_a, "b")
    c = pd.read_csv(I.RESULTS / "core_country.csv")
    share = c[(c.scenario == 47) & (c.bracket == "kept") & (c.year == 3)].set_index("iso3").herd_ni * 100
    ctx = MapContext()
    bounds = [0, 10, 25, 40, 55, 70, 85, 100]
    norm = BoundaryNorm(bounds, 256)
    ctx.choropleth(ax_m, share, "RdYlBu", norm)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu")
    cb = fig.colorbar(sm, ax=ax_m, shrink=0.8, pad=0.01, fraction=0.03, ticks=bounds, label="% of pre-catastrophe")
    ax_m.set_title("$\\bf{c}$  Ruminants alive in year 3 after 47 Tg with grain kept, nothing else done", loc="left", pad=3)
    ctx.finish(ax_m)
    # centre the map and its colour bar, labels included, under panels a and b with theirs
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w = fig.bbox.width
    top = [ax.get_tightbbox(r) for ax in (ax_h, ax_a)]
    low = [ax_m.get_window_extent(r), cb.ax.get_tightbbox(r)]          # the map's frame, not its title or clipped shapes
    shift = ((min(b.x0 for b in top) + max(b.x1 for b in top)) - (min(b.x0 for b in low) + max(b.x1 for b in low))) / 2 / w
    for ax in (ax_m, cb.ax):
        b = ax.get_position()
        ax.set_position([b.x0 + shift, b.y0, b.width, b.height])
    save(fig, "fig5_no_intervention")


if __name__ == "__main__":
    main()
