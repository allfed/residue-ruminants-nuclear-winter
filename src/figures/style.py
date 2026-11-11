"""Shared figure style: one palette, thin marks, recessive axes."""
import os
from src.inputs import FIGURES, WORK

os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"grain": "#eda100", "animal": "#2a78d6", "carcass": "#e34948",
          "ni": "#52514e", "iv": "#2a78d6", "kept": "#2a78d6", "gone": "#eb6834"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SCENARIO_COLORS = {16: "#9bbfe8", 47: "#2a78d6", 150: "#0b3d7a"}
LABELS = {"grain": "grain and bran freed", "animal": "milk and meat change", "carcass": "change in meat from culled animals"}
BRACKET = {"kept": "grain kept", "gone": "grain gone"}
PANEL = ("IND", "USA", "CHN", "BRA")
NAMES = {"IND": "India", "USA": "United States", "BRA": "Brazil", "NZL": "New Zealand", "ETH": "Ethiopia", "CHN": "China", "IRL": "Ireland"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def panel_label(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
