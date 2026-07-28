#!/usr/bin/env python3
"""One-shot visualization of nav_core topological map overlaid on the floor plan."""

import json
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

MAP_FILE = Path(__file__).parent / "office_535_suite21.json"
BACKGROUND_FILE = Path(__file__).parent / "535-mission-map.png"

TAG_PALETTE = {
    "waypoint":   "#00a6ff",
    "red_marker": "#00a6ff",
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


def node_label(node):
    description = node.get("description", "")
    marker_suffix = ", red marker"
    if marker_suffix in description:
        return description.split(marker_suffix, 1)[0]
    return node["name"].replace("_", " ")


def load_image_background(data):
    """Load the annotated PNG, cropped and rotated into the map coordinate frame."""
    if not BACKGROUND_FILE.exists():
        print(f"Background image not found — skipping: {BACKGROUND_FILE}")
        return None

    img = plt.imread(str(BACKGROUND_FILE))
    bbox = data.get("coordinate_system", {}).get("source_floor_bbox_px", {})
    if bbox:
        left = int(bbox["left"])
        top = int(bbox["top"])
        right = int(bbox["right"])
        bottom = int(bbox["bottom"])
        img = img[top:bottom + 1, left:right + 1]

    # Source image orientation matches the map metadata: Up=East, Left=North.
    # Rotate 90° clockwise so East is plot-right and North is plot-up.
    return np.rot90(img, k=-1)


def main():
    with open(MAP_FILE) as f:
        data = json.load(f)

    dims = data["coordinate_system"]["floor_dimensions_m"]
    ew = dims["east_west"]   # x-axis range
    ns = dims["north_south"] # y-axis range

    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(12, 17))

    # --- Image background ---
    bg = load_image_background(data)
    if bg is not None:
        ax.imshow(
            bg,
            extent=[0, ew, 0, ns],
            origin="upper",
            aspect="auto",
            alpha=0.7,
            zorder=0,
        )

    # --- Floor boundary ---
    boundary = data.get("floor_boundary", [])
    if boundary:
        poly = Polygon(boundary, closed=True, fill=False,
                       edgecolor="#2c3e50", linewidth=2.5, zorder=1)
        ax.add_patch(poly)

    # --- Exclusion zones ---
    for zone in data.get("exclusion_zones", []):
        if zone.get("type") == "polygon":
            points = zone.get("points", [])
            if not points:
                continue
            poly = Polygon(
                points,
                closed=True,
                facecolor="#e74c3c",
                alpha=0.22,
                edgecolor="#c0392b",
                linewidth=1.8,
                linestyle="--",
                zorder=2,
            )
            ax.add_patch(poly)
            pts = np.array(points)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            ax.text(cx, cy, f"NO-BOT\n{zone['name'].replace('_', ' ')}",
                    ha="center", va="center",
                    fontsize=6.2, color="#c0392b", fontweight="bold", zorder=10)
            continue

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
        color = "#00a6ff" if traversable else "#e74c3c"
        style = "-" if traversable else ":"
        lw = 2.2 if traversable else 1.0
        alpha = 0.9 if traversable else 0.35
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                color=color, linestyle=style, linewidth=lw, alpha=alpha, zorder=3)

    # --- Nodes ---
    for node in data["nodes"]:
        color = tag_color(node.get("tags", []))
        is_restricted = "restricted" in node.get("tags", [])
        marker = "X" if is_restricted else "o"
        size = 140 if is_restricted else 115
        ax.scatter(node["x"], node["y"], c=color, s=size, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=5)
        ax.annotate(
            textwrap.fill(node_label(node), width=14),
            (node["x"], node["y"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=6.5, color="#102a43", fontweight="bold", zorder=6,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )

    # --- Legend ---
    legend_items = [
        mpatches.Patch(facecolor="#00a6ff", label="Mapped Node / Edge"),
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
    if os.environ.get("SHOW_MAP", "0").strip().lower() in {"1", "true", "yes", "on"}:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
