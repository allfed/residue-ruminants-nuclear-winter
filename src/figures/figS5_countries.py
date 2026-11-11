"""Figure S5: country residue-feed budgets at 47 Tg, grain kept."""
import pandas as pd

from src import inputs as I
from src.figures.style import NAMES, PANEL, SERIES, plt, save


def main():
    c = pd.read_csv(I.RESULTS / "core_country.csv")
    c = c[(c.scenario == 47) & (c.bracket == "kept")]
    fig, axes = plt.subplots(1, len(PANEL), figsize=(7.5, 2.6), sharex=True, layout="constrained")
    for ax, iso in zip(axes, PANEL):
        s = c[c.iso3 == iso].sort_values("year")
        fed_extra = s.residue_extra_fed_pj
        parts = [("usual residue feed eaten", s.residue_fed_pj - fed_extra, SERIES[0]), ("additional residue eaten", fed_extra, SERIES[1]),
                 ("carried to the next year", s.residue_carried_pj, SERIES[2]), ("decayed or not kept", s.residue_decayed_pj, SERIES[3]),
                 ("usual feed left uneaten", s.residue_unused_usual_pj, "#d9d8d3")]
        if (c.residue_unused_usual_pj.abs() < 1).all():             # the intervention offers it as additional residue
            parts = parts[:-1]
        bottom = 0
        for label, v, col in parts:
            ax.bar(s.year, v, 0.75, bottom=bottom, color=col, edgecolor="white", linewidth=0.5, label=label)
            bottom = bottom + v.to_numpy()
        ax.set_ylim(0, 1.1 * bottom.max())                          # the bar's height is the residue that can be fed
        ax.set_title(NAMES[iso])
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("year after the war")
    axes[0].set_ylabel("PJ ME per year")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=len(l), fontsize=6.5)
    save(fig, "figS5_countries")


if __name__ == "__main__":
    main()
