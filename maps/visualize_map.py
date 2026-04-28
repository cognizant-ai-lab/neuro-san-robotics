#!/usr/bin/env python3
"""One-shot visualization of nav_core topological map overlaid on the PDF floor plan."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
import seaborn as sns

MAP_FILE = Path(__file__).parent / "office_535_suite21.json"
PDF_FILE = Path.home() / "Downloads" / "535_21_floor_plan.pdf"

TAG_PALETTE = {
    "entrance":   "#e74c3c",
    "corridor":   "#95a5a6",
    "charging":   "#2ecc71",
    "conference":  "#9b59b6",
    "kitchen":    "#e67e22",
    "lounge":     "#1abc9c",
    "office":     "#3498db",
    "huddle":     "#2980b9",
    "open_plan":  "#f1c40f",
    "workspace":  "#f1c40f",
    "cafe":       "#e67e22",
    "focus":      "#2980b9",
    "stairs":     "#e74c3c",
    "restricted": "#c0392b",
}


def tag_color(tags):
    for t in tags:
        if t in TAG_PALETTE:
            return TAG_PALETTE[t]
    return "#7f8c8d"


def load_pdf_background():
    """Render page 1 of the PDF to a numpy array, rotated 90° CW (Up=East → North-up)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed — skipping PDF background. pip install PyMuPDF")
        return None

    doc = fitz.open(str(PDF_FILE))
    page = doc[0]
    # Rotate 90° clockwise so PDF-Up (East) becomes plot-Right (East)
    # and PDF-Left (North) becomes plot-Up (North)
    mat = fitz.Matrix(2.0, 2.0).prerotate(-90)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    doc.close()
    return img


def main():
    with open(MAP_FILE) as f:
        data = json.load(f)

    dims = data["coordinate_system"]["floor_dimensions_m"]
    ew = dims["east_west"]   # x-axis range
    ns = dims["north_south"] # y-axis range

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 17))

    # --- PDF background ---
    bg = load_pdf_background()
    if bg is not None:
        ax.imshow(bg, extent=[0, ew, 0, ns], aspect="auto", alpha=0.25, zorder=0)

    # --- Floor boundary ---
    boundary = data.get("floor_boundary", [])
    if boundary:
        poly = Polygon(boundary, closed=True, fill=False,
                       edgecolor="#2c3e50", linewidth=2.5, zorder=1)
        ax.add_patch(poly)

    # --- Exclusion zones ---
    for zone in data.get("exclusion_zones", []):
        c = zone["corners"]
        x0, y0 = c[0]
        x1, y1 = c[1]
        buf = zone.get("buffer_m", 0)
        rect = FancyBboxPatch(
            (x0 - buf, y0 - buf), (x1 - x0 + 2 * buf), (y1 - y0 + 2 * buf),
            boxstyle="round,pad=0.1",
            facecolor="#e74c3c", alpha=0.18, edgecolor="#c0392b",
            linewidth=1.8, linestyle="--", zorder=2,
        )
        ax.add_patch(rect)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(cx, cy, f"NO-BOT\n{zone['name'].replace('_', ' ')}",
                ha="center", va="center",
                fontsize=7, color="#c0392b", fontweight="bold", zorder=10)

    # --- Edges ---
    node_lut = {n["name"]: n for n in data["nodes"]}
    for edge in data["edges"]:
        a = node_lut.get(edge["from"])
        b = node_lut.get(edge["to"])
        if not a or not b:
            continue
        traversable = edge.get("traversable", True)
        color = "#7f8c8d" if traversable else "#e74c3c"
        style = "-" if traversable else ":"
        lw = 1.6 if traversable else 1.0
        alpha = 0.6 if traversable else 0.35
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                color=color, linestyle=style, linewidth=lw, alpha=alpha, zorder=3)

    # --- Nodes ---
    for node in data["nodes"]:
        color = tag_color(node.get("tags", []))
        is_restricted = "restricted" in node.get("tags", [])
        marker = "X" if is_restricted else "o"
        size = 140 if is_restricted else 100
        ax.scatter(node["x"], node["y"], c=color, s=size, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            node["name"].replace("_", " "),
            (node["x"], node["y"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=6.5, color="#2c3e50", fontweight="medium", zorder=6,
        )

    # --- Legend ---
    legend_items = [
        mpatches.Patch(facecolor="#95a5a6", label="Corridor / Waypoint"),
        mpatches.Patch(facecolor="#2ecc71", label="Charging Station"),
        mpatches.Patch(facecolor="#3498db", label="Office / Huddle"),
        mpatches.Patch(facecolor="#f1c40f", label="Open Plan / Workspace"),
        mpatches.Patch(facecolor="#e67e22", label="Kitchen / Cafe"),
        mpatches.Patch(facecolor="#1abc9c", label="Lounge"),
        mpatches.Patch(facecolor="#e74c3c", label="Restricted / Entrance"),
        mpatches.Patch(facecolor="#e74c3c", alpha=0.18, edgecolor="#c0392b",
                       linestyle="--", label="Exclusion Zone (NO-BOT)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=7,
              framealpha=0.9, title="Legend", title_fontsize=8)

    # --- Compass ---
    ax.annotate("", xy=(2.0, 3.5), xytext=(2.0, 1.5),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))
    ax.text(2.0, 3.8, "N", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50")
    ax.annotate("", xy=(3.5, 1.5), xytext=(2.0, 1.5),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))
    ax.text(3.8, 1.5, "E", ha="left", va="center",
            fontsize=10, fontweight="bold", color="#2c3e50")

    # --- Formatting ---
    ax.set_xlabel("East  (meters)")
    ax.set_ylabel("North  (meters)")
    ax.set_title(data["name"] + " - Nav Topological Map", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1, ew + 2)
    ax.set_ylim(-1, ns + 2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = MAP_FILE.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
