"""Figure 9: one parameter at a time (tornado) and the structural switches at 47 Tg, ten-year mean of food saved.

main(16) and main(150) draw the same figure for the supplement (Figs. S6, S7)."""
import textwrap

import numpy as np
import pandas as pd

from src import inputs as I
from src.figures.style import BRACKET, COLORS, LABELS, plt, save

# the range goes in the label where one number covers it; residue and feed contents move across each crop's or material's own range
PARAMETER_LABELS = {
    "ndf_cap_large": "Fiber intake limit, cattle and buffaloes, 0.9–1.5% of live weight",
    "ndf_cap_small": "Fiber intake limit, sheep and goats, 1.2–2.0% of live weight",
    "ndf_baseline_factor": "Fiber intake room above the pre-catastrophe ration, 0–20%",
    "rpr_mult": "Residue per unit of crop",
    "residue_me_mult": "Residue energy content", "residue_cp_mult": "Residue protein content",
    "residue_ndf_mult": "Residue fiber content", "protein_mult": "Protein requirements, ±30%",
    "requirement_mult": "Energy requirements, ±30%", "access_radius_km": "Reach radius, 10–200 km",
    "baled_share": "Share baled or stacked, 10–90%", "reserve_other_mult": "Other uses of residue, ±50%",
    "field_decay_factor": "Field decay rate, 40–120%", "grass_me_pos": "Grass energy content", "grass_cp_pos": "Grass protein content",
    "grass_ndf_pos": "Grass fiber content", "conc_me_pos": "Grain and bran energy content", "conc_cp_pos": "Grain and bran protein content",
    "conc_ndf_pos": "Grain and bran fiber content", "coproduct_me_pos": "Co-product energy content",
    "coproduct_cp_pos": "Co-product protein content", "coproduct_ndf_pos": "Co-product fiber content",
    "acclim_rate": "Acclimatization to cold, 0–0.91% per °C",
    "cold_wet": "Share of the cold season wet or windy, 0–100%", "growth_mult": "Herd regrowth rate, 50–200%",
    "k_stored": "Decay in store, 0.3–2.5% per month",
    "clearing_shift": "Next sowing a month earlier or later",
}
# the model choices shown, in two kinds: those that change only the residue offered, against the unchanged run
# without the intervention, and those that change the herds and their feeds in both runs; the others are
# reported in the text (Sections 2.7, 3.4 to 3.6)
SWITCH_GROUPS = {
    "the residue offered": {
        "allocation: dairy first": "Dairy herds fed first",
        "more intense collection (10% left on the field everywhere)": "10% left on the field everywhere",
        "no distance limit within a country": "No distance limit within a country",
        "no distance limit, residue shared between countries": "No distance limit, shared between countries",
        "allocated share reduced by half": "Half the residue within reach",
        "allocated share reduced by three quarters": "A quarter of the residue within reach",
    },
    "the herds and their feeds": {
        "herd strategy: keep the largest herd that survives": "Every animal that can be maintained is kept",
        "herd strategy: most milk and meat eaten each year": "Most milk and meat eaten each year",
        "usual residue fed as a fixed mixture (extra residue still ranked)": "Usual residue eaten as a fixed mixture",
        "no cold stress": "No cold stress",
        "no production in freezing months": "No output in freezing months",
        "oilseed meals and other co-products removed too": "Oilseed meals and other co-products removed",
    },
}
SWITCH_LABELS = {k: v for group in SWITCH_GROUPS.values() for k, v in group.items()}
GROUP_GAP = 0.5          # extra space between the two blocks, in rows


def parameter_label(k, width=34):
    """Wrapped name; the range after the last comma never breaks (no-break spaces), nor does pre-catastrophe at its hyphen."""
    label = PARAMETER_LABELS.get(k, k)
    if ", " in label:
        name, rng = label.rsplit(", ", 1)
        label = f"{name}, {rng.replace(' ', '\u00a0')}"
    return textwrap.fill(label, width, break_on_hyphens=False)


NAMES = {47: "fig9_tornado", 16: "figS6_tornado_16", 150: "figS7_tornado_150"}


def main(tg=47):
    t = pd.read_csv(I.RESULTS / "tornado.csv")
    sw = pd.read_csv(I.RESULTS / "switches.csv")
    fig = plt.figure(figsize=(7.5, 8.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 0.85])
    axes_top = top.subplots(1, 2)
    # c carries the labels for both bottom panels, so d gets no label column of its own
    axes_bottom = bottom.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": [1.0, 0.62]})
    for j, b in enumerate(("kept", "gone")):
        s = t[(t.scenario == tg) & (t.bracket == b)]
        central = float(s[s.kind == "central"].total.iloc[0])
        p = s[s.kind != "central"].pivot(index="label", columns="kind", values="total")
        p["span"] = (p.max(axis=1) - p.min(axis=1))
        p = p.sort_values("span").tail(12)
        ax = axes_top[j]
        y = np.arange(len(p))
        ax.barh(y, p["low"] - central, left=central, color="#9bbfe8", height=0.65, label="parameter at its low value")
        ax.barh(y, p["high"] - central, left=central, color=COLORS["iv"], height=0.65, label="parameter at its high value")
        ax.axvline(central, color="#0b0b0b", lw=0.7, ls=":")
        ax.set_yticks(y); ax.set_yticklabels([parameter_label(k) for k in p.index], fontsize=8)
        ax.set_ylim(-0.6, len(p) - 0.4)
        ax.set_title(f"$\\bf{{{'ab'[j]}}}$  parameters, {BRACKET[b]}", fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        s = sw[(sw.scenario == tg) & (sw.bracket == b)].set_index("label").loc[list(SWITCH_LABELS)]
        if j == 0:
            # rows from the bottom up: each block sorted by the grain-kept total, largest at the top, under its title
            order, rows, titles, yy = [], [], [], 0.0
            for title, group in reversed(list(SWITCH_GROUPS.items())):
                for k in s.loc[list(group)].sort_values("total").index:
                    order.append(k); rows.append(yy); yy += 1
                titles.append((title, yy)); yy += 1 + GROUP_GAP
            rows = np.array(rows)
        s = s.loc[order].copy()              # same order in both panels: the labels are on the left one
        ax = axes_bottom[j]
        y = rows
        pos, neg = np.zeros(len(s)), np.zeros(len(s))
        for component in ("grain", "animal", "carcass"):
            v = s[component].to_numpy()
            ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=COLORS[component], height=0.7, label=LABELS[component])
            pos += np.maximum(v, 0)
            neg += np.minimum(v, 0)
        ax.plot(s.total, y, "o", ms=3.5, color="#0b0b0b", label="total")
        ax.axvline(central, color="#0b0b0b", lw=0.7, ls=":")
        ticks = list(y) + [ty for _, ty in titles]
        ax.set_yticks(ticks)
        if j == 0:
            ax.set_yticklabels([SWITCH_LABELS[k] for k in s.index] + [title for title, _ in titles], fontsize=8)
            for lab in ax.get_yticklabels()[len(s):]:
                lab.set_fontstyle("italic"); lab.set_fontweight("bold")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for tick in ax.yaxis.get_major_ticks()[len(s):]:
            tick.tick1line.set_visible(False)
        ax.set_ylim(-0.6, max(ticks) + 0.6)
        ax.set_xlabel("food saved, kcal per person per day"); ax.grid(axis="y", visible=False)
        lo, hi = min(neg.min(), s.total.min(), 0), max(pos.max(), s.total.max())
        ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
        ax.set_title(f"$\\bf{{{'cd'[j]}}}$  model choices, {BRACKET[b]}", fontsize=9, loc="left")
    top.supxlabel("food saved, kcal per person per day", fontsize=8)
    handles, labels = axes_top[0].get_legend_handles_labels()
    top.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=8)
    handles, labels = axes_bottom[0].get_legend_handles_labels()
    bottom.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    save(fig, NAMES[tg])


if __name__ == "__main__":
    main()
