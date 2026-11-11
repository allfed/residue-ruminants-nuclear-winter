"""Figure 4: where the residues and the ruminants are, and the share of each cell's removable residue (tau e) that the
pre-catastrophe animals within 50 km get, as Fig. 2c (reach/access.py, map ratio mu / tau)."""
import numpy as np
from matplotlib.colors import LogNorm

from src import inputs as I
from src.figures.maps import MapContext
from src.figures.style import plt, save


def main():
    m = np.load(I.DATA / "map_context" / "maps_05deg.npz")
    ctx = MapContext()
    top = float(m["lat_top"])
    grid_x, grid_y = ctx.grid(m["residue_gj"].shape, top)
    ha = m["area_km2"] * 100
    # a and b side by side, c at full width under them; each colour bar centred below its map
    width = 7.5
    aspect = (ctx.bounds[3] - ctx.bounds[2]) / (ctx.bounds[1] - ctx.bounds[0])
    hspace, top, bottom = 0.3, 0.3, 0.3  # gap between rows (share of the mean row), margins in inches
    height = 1.5 * width * aspect * (1 + hspace / 2) + top + bottom
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], left=0, right=1, top=1 - top / height, bottom=bottom / height,
                          hspace=hspace, wspace=0.03)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    # per ha of cell area; the allocated share can exceed one by the flow's rounding in tiny cells, shown as one
    panels = (("Residue energy, GJ per ha", m["residue_gj"] / ha, LogNorm(0.05, 50), "YlGn"),
              ("Ruminant energy needs (all feeds), GJ per ha", m["demand_gj"] / ha, LogNorm(0.05, 50), "Blues"),
              ("Share of available residue allocated to animals within 50 km", np.minimum(m["access50"], 1.0), None, "viridis"))
    for k, (ax, (title, a, norm, cmap)) in enumerate(zip(axes, panels)):
        a = np.where(np.isfinite(a) & (a > 0), a, np.nan)
        ctx.land(ax)
        im = ax.pcolormesh(grid_x, grid_y, a, norm=norm, cmap=cmap, vmin=None if norm else 0, vmax=None if norm else 1,
                           shading="flat", rasterized=True, zorder=1)
        ctx.borders(ax)
        ctx.finish(ax)
        cax = ax.inset_axes([0.35, -0.045, 0.3, 0.025] if k == 2 else [0.3, -0.09, 0.4, 0.05])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        if norm:
            cb.set_ticks([0.1, 1, 10], labels=["0.1", "1", "10"])
            cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
        cb.outline.set_linewidth(0.4)
        ax.set_title(f"$\\bf{{{'abc'[k]}}}$  {title}", loc="left", pad=2)
    save(fig, "fig4_maps")


if __name__ == "__main__":
    main()
