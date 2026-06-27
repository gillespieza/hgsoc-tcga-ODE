"""
plot_hrddr_pathway.py — Static PNG diagram of the HR-DDR ODE pathway.

Produces a publication-quality figure showing the five-variable HR-DDR
mechanistic pathway:
    D(t) : DNA damage load
    A(t) : ATM/ATR checkpoint kinases
    C(t) : CHK1/CHK2 effectors
    R(t) : HR repair complex (BRCA axis)
    X(t) : apoptotic commitment signal

Outputs
-------
- hr_ddr_pathway.png  (300 dpi, ~3000 x 2800 px)

Dependencies
------------
    pip install matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent

# =================================================================
# Colour palette  (matches project: blue/purple/teal/coral/red/amber)
# =================================================================
C = {
    "drug":    "#F2A623",   # amber  — carboplatin input
    "damage":  "#E8593C",   # coral  — DNA damage D(t)
    "atm":     "#7F77DD",   # purple — ATM/ATR A(t)
    "chk":     "#534AB7",   # deep purple — CHK1/2 C(t)
    "hr":      "#1D9E75",   # teal   — HR repair R(t)
    "bcl":     "#888780",   # grey   — BCL2/BAX balance
    "apop":    "#A32D2D",   # red    — apoptosis X(t)
    "bg":      "#FAFAFA",
    "nucleus": "#E8E6F0",
    "mito":    "#E6F5EE",
    "text":    "#1A1A1A",
    "muted":   "#6B6B6B",
    "arrow":   "#444441",
    "inhib":   "#C62828",
    "repair":  "#0F6E56",
}

# =================================================================
# Node geometry helpers
# =================================================================

def draw_box(ax, x, y, w, h, color, title, subtitle=None, rx=0.012):
    """Draw a rounded rectangle node with title and optional subtitle."""
    face = color + "22"   # ~13% opacity fill
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rx}",
        linewidth=1.4,
        edgecolor=color,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(box)

    if subtitle:
        ax.text(x, y + 0.012, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=4)
        ax.text(x, y - 0.022, subtitle, ha="center", va="center",
                fontsize=7.8, color=C["muted"], zorder=4, style="italic")
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=color, zorder=4)


def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=1.4,
          style="-|>", dashed=False, label=None, label_offset=(0, 0)):
    """Draw a straight arrow between two points."""
    ls = (0, (5, 3)) if dashed else "solid"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7, color=C["muted"], zorder=5,
                bbox=dict(fc=C["bg"], ec="none", pad=1.0))


def curved_arrow(ax, x1, y1, x2, y2, color, lw=1.3, dashed=False,
                 rad=0.3, style="-|>", label=None, label_pos=0.5):
    """Draw a curved arrow using FancyArrowPatch."""
    ls = (0, (5, 3)) if dashed else "solid"
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=2,
    )
    ax.add_patch(patch)
    if label:
        # Approximate midpoint on the arc
        mx = (x1 + x2) / 2 - rad * (y2 - y1) * label_pos
        my = (y1 + y2) / 2 + rad * (x2 - x1) * label_pos
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=7, color=C["muted"], zorder=5,
                bbox=dict(fc=C["bg"], ec="none", pad=1.0))


# =================================================================
# Main figure
# =================================================================

def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # -----------------------------------------------------------------
    # Background zones
    # -----------------------------------------------------------------
    # Nucleus
    nucleus = FancyBboxPatch(

        (0.04, 0.06),           # (x, y) position of the bottom-left corner
                                # of the box.

        0.95,                   # Width of the box.

        0.80,                   # Height of the box.

        boxstyle="round,pad=0,rounding_size=0.025",
                                # Shape of the box:
                                # - round          → rounded corners
                                # - pad=0          → no extra padding around contents
                                # - rounding_size  → radius of the rounded corners

        linewidth=1.0,          # Thickness of the border (1 point).

        linestyle=(0, (6, 4)),  # Dashed border:
                                # - offset = 0
                                # - draw 6 points
                                # - gap 4 points
                                # - repeat

        edgecolor="#9490C0",    # Border colour (light purple).

        facecolor=C["nucleus"], # Fill colour from the colour dictionary.

        alpha=0.35,             # Transparency:
                                # 0   = invisible
                                # 1   = fully opaque
                                # 0.35 = 35% opacity

        zorder=0,               # Drawing order.
                                # Lower values are drawn first (appear behind
                                # objects with larger zorder values).
    )
    ax.add_patch(nucleus)
    ax.text(0.07, 0.845, "Nucleus", fontsize=8, color="#9490C0",
            fontstyle="italic", zorder=1)

    # Mitochondrion (lower right)
    mito = FancyBboxPatch(
        (0.62, 0.08), 0.30, 0.22,
        boxstyle="round,pad=0,rounding_size=0.018",
        linewidth=1.0, linestyle=(0, (4, 3)),
        edgecolor=C["hr"], facecolor=C["mito"], alpha=0.45, zorder=0,
    )
    ax.add_patch(mito)
    ax.text(0.645, 0.278, "Mitochondrion", fontsize=7.5, color=C["hr"],
            fontstyle="italic", zorder=1)

    # -----------------------------------------------------------------
    # Nodes  (x, y, width, height)
    # -----------------------------------------------------------------
    # Platinum drug — top centre
    draw_box(
        ax,                     # The matplotlib Axes object to draw onto.

        0.50,                   # x-coordinate of the box (usually in axes coordinates,
                                # where 0 = left edge and 1 = right edge).

        0.925,                  # y-coordinate of the box (usually in axes coordinates,
                                # where 0 = bottom and 1 = top).

        0.26,                   # Width of the box as a fraction of the axes width.

        0.065,                  # Height of the box as a fraction of the axes height.

        C["drug"],              # Fill colour for the box, retrieved from the colour
                                # dictionary using the "drug" key.

        "Carboplatin (Pt)",     # Main label (title) displayed inside the box.

        "Drug input  φ(t) = D₀ · e^(−t/τ)"   # Secondary label or mathematical description
                                            # shown beneath the title.
    )

    # D(t) — DNA damage
    draw_box(ax, 0.50, 0.785, 0.35, 0.065,
             C["damage"], "D(t) — DNA damage load",
             "Double-strand breaks (DSBs) accumulate")

    # A(t) — ATM/ATR
    draw_box(ax, 0.55, 0.640, 0.32, 0.065,
             C["atm"], "A(t) — ATM / ATR kinases",
             "Sense DSBs, activate checkpoint")

    # C(t) — CHK1/2
    draw_box(ax, 0.50, 0.495, 0.32, 0.065,
             C["chk"], "C(t) — CHK1 / CHK2 effectors",
             "Cell cycle arrest, BRCA1 loading")

    # R(t) — HR repair complex (left)
    draw_box(ax, 0.20, 0.640, 0.30, 0.065,
             C["hr"], "R(t) — HR repair complex",
             "BRCA1 · BRCA2 · RAD51 · PALB2")

    # BCL2/BAX balance (lower right)
    draw_box(ax, 0.775, 0.200, 0.26, 0.065,
             C["bcl"], "BCL2 / BAX balance",
             "Anti- vs pro-apoptotic ratio")

    # X(t) — apoptotic commitment (bottom centre-right)
    draw_box(ax, 0.64, 0.115, 0.38, 0.070,
             C["apop"], "X(t) — apoptotic commitment",
             "AUC_X = ∫X dt  →  survival predictor")

    # -----------------------------------------------------------------
    # Arrows
    # -----------------------------------------------------------------
    # Drug → D(t)
    arrow(ax, 0.50, 0.882, 0.50, 0.818,
          color=C["drug"], lw=1.6, label="DSBs", label_offset=(0.025, 0))

    # D(t) → A(t)
    arrow(ax, 0.50, 0.752, 0.50, 0.673,
          color=C["damage"], lw=1.6)

    # A(t) → C(t)
    arrow(ax, 0.50, 0.607, 0.50, 0.528,
          color=C["atm"], lw=1.6)

    # A(t) → R(t)  (rightward to leftward: CHK loads BRCA axis)
    arrow(ax, 0.336, 0.640, 0.353, 0.640,
          color=C["atm"], lw=1.2, dashed=True,
          label="loads HR complex", label_offset=(0, 0.028))

    # C(t) → R(t)  curved: CHK exhausts HR (inhibition)
    curved_arrow(ax, 0.342, 0.495, 0.20, 0.607,
                 color=C["inhib"], lw=1.2, dashed=True,
                 rad=-0.3, style="-[",
                 label="exhausts R\n(k_load·C·R)", label_pos=0.3)

    # R(t) → D(t)  curved: HR repairs damage
    curved_arrow(ax, 0.20, 0.673, 0.352, 0.785,
                 color=C["repair"], lw=1.5,
                 rad=-0.35,
                 label="repairs DSBs", label_pos=0.4)

    # C(t) → X(t)  (apoptotic drive)
    curved_arrow(ax, 0.612, 0.495, 0.680, 0.150,
                 color=C["chk"], lw=1.5, rad=0.25,
                 label="drives X(t)\n(k_x · C)", label_pos=0.45)

    # R(t) → X(t)  HR suppresses apoptosis
    curved_arrow(ax, 0.20, 0.607, 0.454, 0.150,
                 color=C["repair"], lw=1.2, dashed=True,
                 rad=0.15, style="-[",
                 label="suppresses X\n(k_suppress)", label_pos=0.5)

    # BCL2/BAX → X(t)
    arrow(ax, 0.775, 0.167, 0.730, 0.150,
          color=C["bcl"], lw=1.3, label="buffers X\n(d_x·BCL2)", label_offset=(-0.01, 0.03))

    # C(t) → BCL2/BAX  (checkpoint tips apoptotic balance)
    curved_arrow(ax, 0.660, 0.495, 0.775, 0.233,
                 color=C["chk"], lw=1.2, rad=0.18,
                 label="tips balance", label_pos=0.5)

    # -----------------------------------------------------------------
    # Patient-specific parameter annotations (right margin)
    # -----------------------------------------------------------------
    params = [
        (0.67, 0.785, "patient-specific\nexpression"),
        (0.70, 0.615, "ATM_tot = (ATM + ATR) / 2"),
        (0.65, 0.495, "CHK_tot = (CHEK1 + CHEK2) / 2"),
        (0.20, 0.640, "BRCA_cap = f(BRCA1, BRCA2,\nRAD51, PALB2)"),
        (0.775, 0.200, "BCL2_ratio = (BCL2 + BCL2L1)\n/ (BAX + BAD + ε)"),
    ]
    for nx, ny, txt in params:
        ax.annotate(
            txt,
            xy=(nx + 0.001, ny),
            xytext=(0.97, ny),
            xycoords="data", textcoords="data",
            fontsize=6.8, color=C["muted"], ha="right", va="center",
            arrowprops=dict(arrowstyle="-", color=C["muted"],
                            lw=0.7, linestyle=(0, (3, 3))),
            zorder=5,
        )

    # -----------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------
    legend_x, legend_y = 0.06, 0.30
    lw = 0.10
    handles = [
        mpatches.FancyArrow(

            0, 0,                # Starting point of the arrow (x, y).

            1, 0,                # Arrow displacement (dx, dy).
                                 # Starts at (0,0) and extends 1 unit to the right.

            width=0.002,         # Thickness of the arrow shaft.

            color=C["arrow"],    # Arrow colour from the colour dictionary.

            length_includes_head=True,
                                # The specified length (dx = 1) includes
                                # the arrowhead.

            head_width=0.004,    # Width of the arrowhead.
        ),
    ]

    items = [
        (C["arrow"],  "solid",       "Activation / increases"),
        (C["inhib"],  "dashed",      "Inhibits / exhausts (⊣)"),
        (C["repair"], "dashed",      "HR suppresses apoptosis (⊣)"),
    ]
    for i, (col, ls, lab) in enumerate(items):
        y = legend_y - i * 0.035
        lsval = (0, (4, 2)) if ls == "dashed" else "solid"
        ax.plot([legend_x, legend_x + lw], [y, y],
                color=col, lw=1.4, linestyle=lsval, zorder=6)
        if ls == "solid":
            ax.annotate("", xy=(legend_x + lw, y), xytext=(legend_x + lw - 0.001, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4), zorder=6)
        else:
            ax.text(legend_x + lw, y, " ⊣", fontsize=9, color=col,
                    va="center", zorder=6)
        ax.text(legend_x + lw + 0.02, y, lab,
                fontsize=7.5, va="center", color=C["text"], zorder=6)

    # Legend box
    leg_box = FancyBboxPatch(

        (legend_x - 0.015, legend_y - 0.0905),
                                # Bottom-left corner of the legend box.
                                # The offsets position the box slightly to the
                                # left and below the legend's reference point.

        0.35,                    # Width of the legend box.

        0.115,                   # Height of the legend box.

        boxstyle="round,pad=0,rounding_size=0.01",
                                # Rounded rectangle:
                                # - round          → rounded corners
                                # - pad=0          → no extra padding
                                # - rounding_size  → small corner radius

        lw=0.8,                  # Border line width (0.8 points).

        edgecolor=C["muted"],    # Border colour from the colour palette.

        facecolor=C["bg"],       # Background colour of the legend box.

        zorder=5,                # Draw above most other objects.
    )
    ax.add_patch(leg_box)

    # -----------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------
    fig.text(0.5, 0.975,
             "HR-DDR ODE Model — Pathway Diagram",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=C["text"])
    fig.text(0.5, 0.958,
             "HGSOC · Carboplatin chemotherapy · Five-variable mechanistic model",
             ha="center", va="top", fontsize=8.5, color=C["muted"])

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "hr_ddr_pathway.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()