"""
plot_hrddr_pathway.py — Static PNG diagram of the HR-DDR ODE pathway.

Produces a publication-quality figure showing the five-variable HR-DDR
mechanistic pathway:
    D(t) : DNA damage load
    A(t) : ATM/ATR checkpoint kinases
    C(t) : CHK1/CHK2 effectors
    R(t) : HR repair complex (BRCA axis)
    X(t) : apoptotic commitment signal

Key idea:
- Carboplatin input φ(t) drives D(t); checkpoint kinases gate apoptosis
- HR repair R(t) both clears damage and suppresses the apoptotic signal
- AUC_X = ∫X dt is the primary survival predictor

Outputs
-------
- results/figures/hr_ddr_pathway.png  (300 dpi)

Run from the project root:
    python plot_hrddr_pathway.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

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

def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    title: str,
    subtitle: str | None = None,
    *,
    rx: float = 0.012,
    title_fontsize: float = 9.5,
    subtitle_fontsize: float = 7.8,
    title_color: str | None = None,
    subtitle_color: str | None = None,
    title_weight: str = "bold",
    subtitle_style: str = "italic",
    fill_alpha_hex: str = "22",
    edge_lw: float = 1.4,
    zorder_box: int = 3,
    zorder_text: int = 4,
    subtitle_offset: float = 0.022,
    title_offset: float = 0.012,
) -> None:
    """
    Draw a rounded rectangle node with optional subtitle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : float
        Centre position of the node in data coordinates.
    w, h : float
        Width and height of the node.
    color : str
        Border colour; fill is the same colour at low opacity.
    title : str
        Primary label (bold).
    subtitle : str or None
        Optional secondary label below the title.
    rx : float
        Corner rounding radius.
    fill_alpha_hex : str
        Two-character hex suffix appended to `color` for the fill
        (#RRGGBB + AA). "22" ≈ 13% opacity, keeping fills light.
    edge_lw : float
        Border line width.
    zorder_box, zorder_text : int
        Drawing order for the box patch and text elements respectively.
    subtitle_offset, title_offset : float
        Vertical displacement applied when both title and subtitle are present.
    """

    # ----------------------------
    # Box fill colour (hex alpha hack)
    # ----------------------------
    face = color + fill_alpha_hex
    # NOTE:
    # assumes RGBA hex-capable backend (#RRGGBBAA style)

    # ----------------------------
    # Node box
    # ----------------------------
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,

        boxstyle=f"round,pad=0,rounding_size={rx}",
        # rounded node shape

        linewidth=edge_lw,
        # border thickness

        edgecolor=color,
        # node outline colour

        facecolor=face,
        # semi-transparent fill

        zorder=zorder_box,
        # background layer for nodes
    )

    ax.add_patch(box)

    # ----------------------------
    # Text styling defaults
    # ----------------------------
    title_color = title_color or color
    subtitle_color = subtitle_color or C["muted"]

    # ----------------------------
    # Text placement
    # ----------------------------
    if subtitle:

        ax.text(
            x,
            y + title_offset,
            title,

            ha="center",
            va="center",

            fontsize=title_fontsize,
            fontweight=title_weight,
            color=title_color,

            zorder=zorder_text,
        )

        ax.text(
            x,
            y - subtitle_offset,
            subtitle,

            ha="center",
            va="center",

            fontsize=subtitle_fontsize,
            color=subtitle_color,

            style=subtitle_style,

            zorder=zorder_text,
        )

    else:

        ax.text(
            x,
            y,
            title,

            ha="center",
            va="center",

            fontsize=title_fontsize,
            fontweight=title_weight,
            color=title_color,

            zorder=zorder_text,
        )


def arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = C["arrow"],
    *,
    lw: float = 1.4,
    style: str = "-|>",
    dashed: bool = False,
    label: str | None = None,
    label_offset: tuple[float, float] = (0, 0),
    label_fontsize: float = 7,
    label_color: str | None = None,
    label_bbox: bool = True,
    bbox_fc: str | None = None,
    bbox_ec: str = "none",
    bbox_pad: float = 1.0,
    zorder: int = 2,
) -> None:
    """
    Draw a straight arrow between two points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x1, y1 : float
        Start point in data coordinates.
    x2, y2 : float
        End point in data coordinates.
    color : str
        Arrow colour.
    lw : float
        Arrow shaft line width.
    style : str
        Arrowhead style string passed to annotate.
    dashed : bool
        Use a dashed linestyle for inhibitory interactions.
    label : str or None
        Optional text placed at the midpoint of the arrow.
    label_offset : (float, float)
        (dx, dy) applied to the midpoint label position.
    label_fontsize : float
        Font size of the midpoint label.
    label_color : str or None
        Overrides default label colour.
    label_bbox : bool
        Draw a background box behind the label.
    bbox_fc, bbox_ec : str
        Label box fill and edge colours.
    bbox_pad : float
        Padding around the label text.
    zorder : int
        Drawing order.
    """

    # ----------------------------
    # Line style
    # ----------------------------
    linestyle = (0, (5, 3)) if dashed else "solid"

    # ----------------------------
    # Arrow
    # ----------------------------
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),

        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            linestyle=linestyle,
            connectionstyle="arc3,rad=0.0",
            # straight line (no curvature)
        ),

        zorder=zorder,
    )

    # ----------------------------
    # Optional label
    # ----------------------------
    if label:

        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]

        ax.text(
            mx,
            my,
            label,

            ha="center",
            va="center",

            fontsize=label_fontsize,
            color=label_color or C["muted"],

            zorder=zorder + 3,

            bbox=(
                dict(
                    fc=bbox_fc if bbox_fc is not None else C["bg"],
                    ec=bbox_ec,
                    pad=bbox_pad,
                )
                if label_bbox
                else None
            ),
        )


def curved_arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    *,
    lw: float = 1.3,
    dashed: bool = False,
    rad: float = 0.3,
    style: str = "-|>",
    label: str | None = None,
    label_pos: float = 0.5,
    label_fontsize: float = 7,
    label_color: str | None = None,
    label_bbox: bool = True,
    bbox_fc: str | None = None,
    bbox_ec: str = "none",
    bbox_pad: float = 1.0,
    zorder: int = 2,
) -> FancyArrowPatch:
    """
    Draw a curved arrow between two points using FancyArrowPatch.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x1, y1 : float
        Start point in data coordinates.
    x2, y2 : float
        End point in data coordinates.
    color : str
        Arrow colour.
    lw : float
        Arrow shaft line width.
    dashed : bool
        Use a dashed linestyle for inhibitory interactions.
    rad : float
        Arc curvature: positive and negative values arc in opposite
        directions; larger magnitudes produce more pronounced curves.
    style : str
        Arrowhead style (e.g. "-|>" for activation, "-[" for inhibition).
    label : str or None
        Optional text placed near the midpoint of the curve.
    label_pos : float
        Controls label displacement along the curvature direction.
    label_fontsize : float
        Font size of the label.
    label_color : str or None
        Overrides default label colour.
    label_bbox : bool
        Draw a background box behind the label.
    bbox_fc, bbox_ec : str
        Label box fill and edge colours.
    bbox_pad : float
        Padding around the label text.
    zorder : int
        Drawing order.

    Returns
    -------
    FancyArrowPatch
        The added patch, for optional further styling by the caller.
    """

    # ----------------------------
    # Line style selection
    # ----------------------------
    linestyle = (0, (5, 3)) if dashed else "solid"

    # ----------------------------
    # Curved arrow
    # ----------------------------
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),

        connectionstyle=f"arc3,rad={rad}",
        # curvature control:
        # rad > 0 → curve one direction
        # rad < 0 → curve opposite direction (flipped)
        # larger |rad| values produce a more pronounced curve.

        arrowstyle=style,
        # arrow head type

        color=color,
        # arrow colour

        linewidth=lw,
        # thickness of arrow shaft

        linestyle=linestyle,
        # dashed or solid interaction type

        zorder=zorder,
        # layer ordering in plot
    )

    ax.add_patch(patch)

    # ----------------------------
    # Optional label
    # ----------------------------
    if label:

        mx = (x1 + x2) / 2 - rad * (y2 - y1) * label_pos
        my = (y1 + y2) / 2 + rad * (x2 - x1) * label_pos
        # heuristic midpoint adjustment:
        # approximates curvature displacement for label placement

        ax.text(
            mx,
            my,
            label,

            ha="center",
            va="center",

            fontsize=label_fontsize,
            color=label_color or color,

            zorder=zorder + 3,

            bbox=(
                dict(
                    fc=bbox_fc if bbox_fc is not None else C["bg"],
                    ec=bbox_ec,
                    pad=bbox_pad,
                )
                if label_bbox
                else None
            ),
        )

    return patch


# =================================================================
# Main
# =================================================================

def main() -> None:
    """
    Render and save the HR-DDR pathway diagram to results/figures/.

    The figure is a static publication-quality layout showing all five ODE
    state variables, their interactions, and the patient-specific gene
    expression parameters that feed into each node. Compartment backgrounds
    (nucleus, mitochondrion) are drawn first so all network elements sit on
    top.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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

        (0.04, 0.06),
        # (x, y): bottom-left corner of nucleus region in axes/data coordinates

        0.95,
        # width: spans almost full diagram width

        0.80,
        # height: large vertical region covering most nuclear processes

        boxstyle="round,pad=0,rounding_size=0.025",
        # box styling:
        # - round: enables rounded rectangle
        # - pad=0: no internal padding around box boundary
        # - rounding_size: controls curvature radius of corners

        linewidth=1.0,
        # border thickness (moderate visibility without overpowering diagram)

        linestyle=(0, (6, 4)),
        # dashed border pattern:
        # - 6 units visible line
        # - 4 units gap
        # creates soft compartment boundary style

        edgecolor="#9490C0",
        # nucleus boundary colour (soft purple tone independent of palette dict)

        facecolor=C["nucleus"],
        # fill colour representing nuclear compartment background

        alpha=0.35,
        # transparency of nuclear region background:
        # lower values increase visibility of underlying network nodes

        zorder=0,
        # background layer (drawn behind all biological entities)
    )

    ax.add_patch(nucleus)

    ax.text(
        0.07, 0.845,
        "Nucleus",
        # label identifying nuclear compartment

        fontsize=8,
        # small label size to avoid competing with node labels

        color="#9490C0",
        # matches nucleus border colour for visual grouping

        fontstyle="italic",
        # stylistic cue for compartment annotation

        zorder=1,
        # slightly above background box but below network elements
    )


    # Mitochondrion (lower right)
    mito = FancyBboxPatch(

        (0.62, 0.08),
        # bottom-left position of mitochondrial region (lower-right quadrant)

        0.30,
        # width of mitochondrial compartment

        0.22,
        # height of mitochondrial compartment

        boxstyle="round,pad=0,rounding_size=0.018",
        # slightly tighter rounding than nucleus for visual distinction

        linewidth=1.0,
        # consistent border thickness across compartments

        linestyle=(0, (4, 3)),
        # shorter dashed pattern:
        # distinguishes mitochondrial boundary from nucleus boundary style

        edgecolor=C["hr"],
        # edge colour tied to homologous recombination / repair palette tone

        facecolor=C["mito"],
        # mitochondrial background fill colour from palette

        alpha=0.45,
        # slightly more opaque than nucleus to emphasise locality

        zorder=0,
        # keeps compartment behind all biological nodes and edges
    )

    ax.add_patch(mito)

    ax.text(
        0.645, 0.278,
        "Mitochondrion",
        # compartment label

        fontsize=7.5,
        # slightly smaller than nucleus label for visual hierarchy

        color=C["hr"],
        # colour aligned with mitochondrial/repair pathway theme

        fontstyle="italic",
        # italic styling for compartment annotation consistency

        zorder=1,
        # above background but below network elements
    )

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
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.50,                   # x-coordinate of the box centre
        0.785,                  # y-coordinate of the box centre
        0.35,                   # Width of the node
        0.065,                  # Height of the node
        C["damage"],            # Border colour (fill is a transparent version)
        "D(t) — DNA damage load",   # Main title
        "Double-strand breaks (DSBs) accumulate",  # Subtitle
    )

    # A(t) — ATM/ATR
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.55,                   # x-coordinate of the box centre
        0.640,                  # y-coordinate of the box centre
        0.32,                   # Width of the node
        0.065,                  # Height of the node
        C["atm"],               # Border colour
        "A(t) — ATM / ATR kinases",  # Main title
        "Sense DSBs, activate checkpoint",  # Subtitle
    )

    # C(t) — CHK1/2
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.50,                   # x-coordinate of the box centre
        0.495,                  # y-coordinate of the box centre
        0.32,                   # Width of the node
        0.065,                  # Height of the node
        C["chk"],               # Border colour
        "C(t) — CHK1 / CHK2 effectors",  # Main title
        "Cell cycle arrest, BRCA1 loading",  # Subtitle
    )

    # R(t) — HR repair complex (left)
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.20,                   # x-coordinate of the box centre
        0.640,                  # y-coordinate of the box centre
        0.30,                   # Width of the node
        0.065,                  # Height of the node
        C["hr"],                # Border colour
        "R(t) — HR repair complex",  # Main title
        "BRCA1 · BRCA2 · RAD51 · PALB2",  # Subtitle
    )

    # BCL2/BAX balance (lower right)
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.775,                  # x-coordinate of the box centre
        0.200,                  # y-coordinate of the box centre
        0.26,                   # Width of the node
        0.065,                  # Height of the node
        C["bcl"],               # Border colour
        "BCL2 / BAX balance",   # Main title
        "Anti- vs pro-apoptotic ratio",  # Subtitle
    )

    # X(t) — apoptotic commitment (bottom centre-right)
    draw_box(
        ax,                     # Matplotlib Axes object to draw onto
        0.34,                   # x-coordinate of the box centre
        0.105,                  # y-coordinate of the box centre
        0.38,                   # Width of the node
        0.070,                  # Height of the node
        C["apop"],              # Border colour
        "X(t) — apoptotic commitment",  # Main title
        "AUC_X = ∫X dt  →  survival predictor",  # Subtitle
    )

    # -----------------------------------------------------------------
    # Arrows
    # -----------------------------------------------------------------
    # Drug → D(t)
    arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.50, 0.882,            # Start point (x1, y1)
        0.50, 0.818,            # End point (x2, y2)
        color=C["drug"],        # Arrow colour
        lw=1.6,                 # Line width
        label="DSBs",           # Label displayed near the arrow
        label_offset=(0.025, 0) # (x, y) offset applied to the label position
    )

    # D(t) → A(t)
    arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.50, 0.752,            # Start point (x1, y1)
        0.50, 0.673,            # End point (x2, y2)
        color=C["damage"],      # Arrow colour
        lw=1.6,                 # Line width
    )

    # A(t) → C(t)
    arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.50, 0.607,            # Start point (x1, y1)
        0.50, 0.528,            # End point (x2, y2)
        color=C["atm"],         # Arrow colour
        lw=1.6,                 # Line width
    )

    # A(t) → R(t) (rightward to leftward: CHK loads BRCA axis)
    arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.336, 0.640,           # Start point (x1, y1)
        0.353, 0.640,           # End point (x2, y2)
        color=C["atm"],         # Arrow colour
        lw=1.2,                 # Line width
        dashed=True,            # Draw as a dashed line
        label="loads HR complex",   # Label displayed near the arrow
        label_offset=(0, 0.028) # Vertical offset for the label
    )

    # C(t) → R(t) curved: CHK exhausts HR (inhibition)
    curved_arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.342, 0.495,           # Start point (x1, y1)
        0.20, 0.607,            # End point (x2, y2)
        color=C["inhib"],       # Arrow colour
        lw=1.2,                 # Line width
        dashed=True,            # Draw a dashed arrow
        rad=-0.3,               # Curve direction and strength
                                # Negative values flip the curve
        style="-[",             # Bar-head indicating inhibition (⊣)
        label="exhausts R\n(k_load·C·R)",  # Label text
        label_pos=0.3,          # Position of the label along the curve
    )

    # R(t) → D(t) curved: HR repairs damage
    curved_arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.20, 0.673,            # Start point (x1, y1)
        0.352, 0.785,           # End point (x2, y2)
        color=C["repair"],      # Arrow colour
        lw=1.5,                 # Line width
        rad=-0.35,              # Curve direction and amount
        label="repairs DSBs",   # Label displayed along the curve
        label_pos=0.4,          # Position of the label on the curve
    )

    # C(t) → X(t) (apoptotic drive)
    # rad controls the direction and amount of curvature:
    #   +ve = curve one way
    #   -ve = curve the opposite way (flipped)
    #   Larger |rad| values produce a more pronounced curve.
    curved_arrow(
        ax,
        0.512, 0.465,          # Start point (x1, y1)
        0.480, 0.135,          # End point (x2, y2)
        color=C["chk"],        # Arrow colour
        lw=1.5,                # Line width
        rad=-0.25,             # Negative value flips the curve
        label="drives X(t)\n(k_x · C)",  # Label displayed along the arrow
        label_pos=0.25,        # Position of the label (0 = start, 1 = end)
    )

    # R(t) → X(t) HR suppresses apoptosis
    curved_arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.20, 0.607,            # Start point (x1, y1)
        0.454, 0.150,           # End point (x2, y2)
        color=C["repair"],      # Arrow colour
        lw=1.2,                 # Line width
        dashed=True,            # Draw a dashed arrow
        rad=0.15,               # Curve direction and amount
        style="-[",             # Bar-head indicating inhibition (⊣)
        label="suppresses X\n(k_suppress)",   # Label displayed along the curve
        label_pos=0.5,          # Midpoint of the curve
    )

    # BCL2/BAX → X(t)
    arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.675, 0.167,           # Start point (x1, y1)
        0.530, 0.100,           # End point (x2, y2)
        color=C["bcl"],         # Arrow colour
        lw=1.3,                 # Line width
        label="buffers X\n(d_x·BCL2)",   # Label displayed near the arrow
        label_offset=(-0.01, 0.03)       # Offset applied to the label
    )

    # C(t) → BCL2/BAX (checkpoint tips apoptotic balance)
    curved_arrow(
        ax,                     # Matplotlib Axes object to draw onto
        0.660, 0.495,           # Start point (x1, y1)
        0.775, 0.233,           # End point (x2, y2)
        color=C["chk"],         # Arrow colour
        lw=1.2,                 # Line width
        rad=0.18,               # Curve direction and amount
        label="tips balance",   # Label displayed along the curve
        label_pos=0.5,          # Midpoint of the curve
    )

    # -----------------------------------------------------------------
    # Patient-specific parameter annotations (right margin)
    # -----------------------------------------------------------------

    params = [
        (0.67, 0.785, "patient-specific\nexpression"),
        # (nx, ny, text)
        # nx, ny: data-coordinates where the annotation originates (source point in diagram)
        # text: label shown on the right margin with line break support

        (0.70, 0.615, "ATM_tot = (ATM + ATR) / 2"),
        # ATM aggregate definition:
        # combines ATM and ATR into a mean "total ATM activity" proxy

        (0.65, 0.495, "CHK_tot = (CHEK1 + CHEK2) / 2"),
        # CHK1/CHK2 combined checkpoint kinase signal (averaged)

        (0.20, 0.640, "BRCA_cap = f(BRCA1, BRCA2,\nRAD51, PALB2)"),
        # functional BRCA pathway capacity derived from key homologous recombination genes
        # explicit multiline label (\n) used for readability in diagram
    ]

    for nx, ny, txt in params:

        ax.annotate(
            txt,

            xy=(nx + 0.001, ny),
            # xy: anchor point near the biological node in the diagram
            # small +0.001 shift prevents overlap with node edge

            xytext=(0.97, ny),
            # xytext: fixed right-margin position (creates consistent label column)

            xycoords="data",
            # xy uses data coordinate system (matches plotted network layout)

            textcoords="data",
            # annotation text is also placed in data coordinates (not axes-fraction space)

            fontsize=6.8,
            # small font to prevent crowding in margin annotation area

            color=C["muted"],
            # muted colour ensures these labels remain secondary information

            ha="right",
            # right-aligned text so labels "hang" neatly from margin edge

            va="center",
            # vertically centred with corresponding node level

            arrowprops=dict(
                arrowstyle="-",                  # simple line pointer (no arrow head)
                color=C["muted"],                # same colour as annotation text
                lw=0.7,                          # thin connector line
                linestyle=(0, (3, 3))            # dashed pattern (3 on, 3 off)
            ),

            zorder=5,
            # ensures annotations sit above base diagram but below key highlights
        )

    # BCL2/BAX annotation: L-shaped connector that drops below the box then
    # runs right to the margin label column.  Going underneath keeps the line
    # clear of the mitochondrion background and the arrows above the node.
    bcl_box_right  = 0.905   # right edge of the BCL2/BAX node (centre 0.775 + w/2 0.13)
    bcl_box_bottom = 0.1675  # bottom edge of the BCL2/BAX node (centre 0.200 - h/2 0.0325)
    bcl_label_y    = 0.129   # elbow y-level: below the box, clear of the node border
    bcl_label_x    = 0.87    # right-margin label column, consistent with other annotations

    # Vertical leg: drops from the bottom-right corner of the box down to the elbow.
    ax.plot(
        [bcl_box_right, bcl_box_right],   # constant x — vertical segment
        [bcl_box_bottom, bcl_label_y],    # from box bottom edge down to elbow y
        color=C["muted"],
        lw=0.7,
        linestyle=(0, (3, 3)),            # dashed pattern (3 on, 3 off)
        zorder=5,
    )

    # Horizontal leg: runs from the elbow rightward to the label column.
    ax.plot(
        [bcl_box_right, bcl_label_x],    # from elbow x across to label column
        [bcl_label_y,   bcl_label_y],    # constant y — horizontal segment
        color=C["muted"],
        lw=0.7,
        linestyle=(0, (3, 3)),            # dashed pattern (3 on, 3 off)
        zorder=5,
    )

    ax.text(
        bcl_label_x,
        bcl_label_y,
        "BCL2_ratio = (BCL2 + BCL2L1)\n/ (BAX + BAD + ε)",
        # apoptotic balance ratio:
        # anti-apoptotic (BCL2, BCL2L1) vs pro-apoptotic (BAX, BAD)
        # epsilon added to avoid division by zero

        ha="right",
        # right-aligned text so labels "hang" neatly from margin edge

        va="center",
        # vertically centred with corresponding node level

        fontsize=6.8,
        # small font to prevent crowding in margin annotation area

        color=C["muted"],
        # muted colour ensures these labels remain secondary information

        zorder=5,
        # ensures annotations sit above base diagram but below key highlights
    )

    # -----------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------

    legend_x, legend_y = 0.06, 0.30   # base anchor position for legend (axes coordinates)
    lw = 0.10                         # length of legend line segment

    items = [
        (C["arrow"],  "solid",  "Activation / increases"),      # activating interaction (solid line)
        (C["inhib"],  "dashed", "Inhibits / exhausts (⊣)"),     # inhibitory interaction
        (C["repair"], "dashed", "HR suppresses apoptosis (⊣)"), # HR-mediated suppression effect
    ]

    for i, (col, ls, lab) in enumerate(items):

        y = legend_y - i * 0.035
        # vertical spacing between legend entries

        lsval = (0, (4, 2)) if ls == "dashed" else "solid"
        # converts label into matplotlib linestyle:
        # dashed → dash pattern, solid → continuous line

        ax.plot(
            [legend_x, legend_x + lw],  # start and end x positions of legend line
            [y, y],                     # constant y (horizontal line)
            color=col,                  # line colour per legend item
            lw=1.4,                     # line thickness
            linestyle=lsval,            # solid or dashed style
            zorder=6                    # draw above most diagram elements
        )

        if ls == "solid":
            ax.annotate(
                "",
                xy=(legend_x + lw, y),           # arrow tip position
                xytext=(legend_x + lw - 0.001, y),
                # tiny offset start point (creates arrow head)
                arrowprops=dict(
                    arrowstyle="-|>",             # arrow head style
                    color=col,                    # arrow colour
                    lw=1.4                        # arrow line width
                ),
                zorder=6
            )
        else:
            ax.text(
                legend_x + lw, y,                # position at end of line
                " ⊣",                            # inhibition symbol
                fontsize=9,                      # symbol size
                color=col,                       # match interaction colour
                va="center",                     # vertical alignment centred on line
                zorder=6
            )

        ax.text(
            legend_x + lw + 0.02, y,             # label position offset from line end
            lab,                                 # legend text label
            fontsize=7.5,                        # label font size
            va="center",                         # vertically centred
            color=C["text"],                     # primary text colour
            zorder=6
        )

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

    fig.text(
        0.5, 0.975,
        "HR-DDR ODE Model — Pathway Diagram",
        ha="center",              # horizontal alignment: centered on x-position
        va="top",                 # vertical alignment: align top of text to y-position
        fontsize=13,              # main title font size
        fontweight="bold",        # makes title bold for emphasis
        color=C["text"]           # primary text colour from palette dictionary
    )

    fig.text(
        0.5, 0.958,
        "HGSOC · Carboplatin chemotherapy · Five-variable mechanistic model",
        ha="center",              # horizontally centered under main title
        va="top",                 # vertically aligned to top anchor point
        fontsize=8.5,             # smaller subtitle font size
        color=C["muted"]          # muted secondary colour for subtitle
    )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "hr_ddr_pathway.png"
    fig.savefig(
        out_path, dpi=300, bbox_inches="tight",
        facecolor=C["bg"], edgecolor="none",
    )
    plt.close(fig)
    logger.info(f"[FILE] Saved: hr_ddr_pathway.png")


if __name__ == "__main__":
    main()