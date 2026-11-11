"""Figure 8: what limits residue feeding. Fate of the residue at 47 Tg and the binding constraint per country (Figs. S3 and S4 at 16 and 150 Tg)."""
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src import inputs as I
from src.figures.maps import MapContext
from src.figures.style import BRACKET, SERIES, panel_label, plt, save

ORDER = ["none", "regrowth limit", "supply", "intake", "protein", "herd gone", "no herd"]
NAMES = ["none: full pre-catastrophe output", "regrowth limit: full output per head, herd not yet rebuilt", "feed supply",
         "intake: rumen fill", "protein", "herd gone", "no herd"]
FILES = {47: "fig8_limits", 16: "figS3_limits_16", 150: "figS4_limits_150"}
CONSTRAINT_COLORS = ["#d9d8d3", "#9bbfe8", SERIES[0], SERIES[1], SERIES[2], "#0b0b0b", "#f4f3ef"]


def main(tg=47):
    f = pd.read_csv(I.RESULTS / "core_fate.csv")
    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.5], width_ratios=[1, 1, 0.62], hspace=0.5, wspace=0.12)
    top = [fig.add_subplot(gs[0, 0])]
    top.append(fig.add_subplot(gs[0, 1], sharey=top[0]))
    key = fig.add_subplot(gs[0, 2]); key.axis("off")
    for k, (ax, b) in enumerate(zip(top, ("kept", "gone"))):
        s = f[(f.scenario == tg) & (f.bracket == b)]
        parts = [("eaten within the usual ration", s.fed - s.fed_extra, SERIES[0]), ("eaten from the additional pool", s.fed_extra, SERIES[1]),
                 ("carried to the next year", s.carried, SERIES[2]), ("decayed or not kept", s.decayed, SERIES[3]),
                 ("usual feed left uneaten", s.unused_usual, "#d9d8d3")]
        if (f.unused_usual.abs() < 1e-3).all():                     # the intervention offers it as additional residue
            parts = parts[:-1]
        bottom = 0
        for label, v, col in parts:
            ax.bar(s.year, v, 0.75, bottom=bottom, color=col, edgecolor="white", linewidth=0.5, label=label)
            bottom = bottom + v.to_numpy()
        ax.set_title(f"{tg} Tg, {BRACKET[b]}")
        ax.set_xlabel("year after the war")
        ax.set_xticks(range(1, 11))
        panel_label(ax, "ab"[k])
    top[0].set_ylabel("EJ of metabolizable energy per year")
    plt.setp(top[1].get_yticklabels(), visible=False)
    h, l = top[0].get_legend_handles_labels()
    key.legend(h, l, loc="center left", fontsize=6.8, ncol=1)
    ax = fig.add_subplot(gs[1, :])
    b = pd.read_csv(I.RESULTS / "core_binding.csv")
    b = b[(b.scenario == tg) & (b.bracket == "gone")].set_index("iso3").dominant
    code = b.map({k: i for i, k in enumerate(ORDER)}).dropna()
    ctx = MapContext()
    ctx.choropleth(ax, code, ListedColormap(CONSTRAINT_COLORS), plt.Normalize(-0.5, len(ORDER) - 0.5))
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(CONSTRAINT_COLORS, NAMES)], loc="upper center",
              bbox_to_anchor=(0.5, -0.01), fontsize=6.5, ncol=4, handlelength=1.2, columnspacing=1.2)
    ax.set_title(f"$\\bf{{c}}$  What limits production (and residue feeding), year 2 of the {tg} Tg scenario, grain gone", loc="left", pad=3)
    ctx.finish(ax)
    save(fig, FILES[tg])


if __name__ == "__main__":
    main()
