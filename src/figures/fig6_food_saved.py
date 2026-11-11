"""Figure 6: food saved per person and day, by year and component, with the Monte Carlo interval."""
import numpy as np
import pandas as pd

from src import inputs as I
from src.figures.style import BRACKET, COLORS, LABELS, plt, save


def stacked(ax, x, parts, width=0.7):
    pos, neg = np.zeros(len(x)), np.zeros(len(x))
    for k, v in parts.items():
        v = np.asarray(v)
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, width, bottom=base, color=COLORS[k], edgecolor="white", linewidth=0.6, label=LABELS[k])
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)


def main():
    w = pd.read_csv(I.RESULTS / "core_world.csv")
    w["animal"] = w.milk + w.meat
    mc = pd.read_csv(I.RESULTS / "mc_world.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.4), sharex=True, sharey="row")
    for i, b in enumerate(("kept", "gone")):
        for j, tg in enumerate(I.SCENARIOS):
            ax = axes[i, j]
            s = w[(w.scenario == tg) & (w.bracket == b)]
            stacked(ax, s.year, {k: s[k] for k in ("grain", "animal", "carcass")})
            m = mc[(mc.scenario == tg) & (mc.bracket == b) & (mc.component == "total") & (mc.year > 0)].sort_values("year")
            ax.errorbar(m.year, (m["p2.5"] + m["p97.5"]) / 2, yerr=(m["p97.5"] - m["p2.5"]) / 2, fmt="none", ecolor="#0b0b0b",
                        lw=0.7, capsize=1.5, label="total: 95% interval of the draws")
            ax.plot(s.year, s.grain + s.animal + s.carcass, "D", ms=3.2, mfc="white", mec="#0b0b0b", mew=0.8, ls="none",
                    label="total, central run")
            ax.axhline(0, color="#52514e", lw=0.6)
            if i == 0:
                ax.set_title(f"{tg} Tg")
            if j == 0:
                ax.set_ylabel(f"{BRACKET[b]}: food saved\nkcal per person per day")
            if i == 1:
                ax.set_xlabel("year after the war")
            ax.set_xticks(range(1, 11))
    h, l = axes[1, 0].get_legend_handles_labels()
    order = [l.index(k) for k in (LABELS["grain"], LABELS["animal"], LABELS["carcass"], "total, central run", "total: 95% interval of the draws")]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False)
    save(fig, "fig6_food_saved")


if __name__ == "__main__":
    main()
