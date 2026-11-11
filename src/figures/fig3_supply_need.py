"""Figure 3: residue energy produced and feedable against what the pre-catastrophe herds need.

Everything is at the pre-catastrophe herds, so the figure does not depend on the grain case or on
how the herds fare. The bars stack the residue that can be fed: the usual rations (prep.pool_usual;
year 0, what the herds fed before the catastrophe), the additional residue of the year (prep.pool less the usual
rations) and, in year 1, the stock on the fields at the start (prep.stock0). The stock carried in
later years depends on what the herds leave uneaten and is not shown."""
import numpy as np

from src import inputs as I
from src.figures.style import SERIES, plt, save
from src.model.mechanism import efficiencies
from src.model.prepare import prepare


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6), sharex=True, sharey=True)
    years = np.arange(0, I.N_YEARS + 1)
    for j, tg in enumerate(I.SCENARIOS):
        prep = prepare(tg)
        x = prep.extras
        km = efficiencies(x["q0"])[0]
        maint = (prep.n0[:, :, None] * prep.ne_m[:, :, None] * prep.cold / km[:, :, None] * 365).sum(axis=(0, 1)) / 1e12
        prewar = np.full(len(years), prep.e0.sum() / 1e12)
        produced = x["e_res"].sum(axis=(0, 2)) / 1e12
        usual = np.concatenate([[x["base_crop_res"].sum() / 1e12], prep.pool_usual.sum(axis=(0, 2)) / 1e12])
        new = np.concatenate([[0.0], (prep.pool - prep.pool_usual).sum(axis=(0, 2)) / 1e12])
        stock = np.zeros(len(years))
        stock[1] = prep.stock0.sum() / 1e12
        ax = axes[j]
        ax.bar(years, usual, width=0.7, color="#7fcfb0", label="in the usual rations")
        ax.bar(years, new, width=0.7, bottom=usual, color=SERIES[2], label="additional residue of the year")
        ax.bar(years, stock, width=0.7, bottom=usual + new, color=SERIES[3], label="stock on the fields at the start")
        ax.plot(years, prewar, color=SERIES[7], ls=":", label="eaten before the catastrophe")
        ax.plot(years, maint, color=SERIES[7], label="maintenance, with cold")
        ax.plot(years, produced, color=SERIES[0], label="produced")
        ax.set_title(f"{tg} Tg")
        ax.set_xlabel("year after the war\n(0 = pre-catastrophe)")
        ax.set_ylim(0, None); ax.set_xticks(range(0, 11))
    axes[0].set_ylabel("EJ of metabolizable energy per year")
    h, l = axes[0].get_legend_handles_labels()                     # lines (intake, maintenance, produced), then bars
    groups = (("Crop residue", (2,)), ("Crop residue that can be fed", (3, 4, 5)), ("Pre-catastrophe herds, all feeds", (0, 1)))
    for ax, (title, idx) in zip(axes, groups):                      # one block under each panel, left-aligned with it
        x0 = ax.get_position().x0
        fig.legend([h[i] for i in idx], [l[i] for i in idx], title=title, loc="upper left", bbox_to_anchor=(x0 - 0.01, -0.14),
                   alignment="left", title_fontproperties={"weight": "bold", "size": 8}, borderpad=0, borderaxespad=0,
                   labelspacing=0.35, handlelength=1.8)
    save(fig, "fig3_supply_need")


if __name__ == "__main__":
    main()
