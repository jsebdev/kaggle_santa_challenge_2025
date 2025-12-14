from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from utils.tree import ChristmasTree
from utils.color_map import get_colors_for_trees


def get_artists_for_configuration(axis, colors):
    tree_outlines = []
    tree_fills = []
    for i in range(len(colors)):
        (outline,) = axis.plot([], [], color=colors[i], linewidth=1)
        fill = axis.fill([], [], alpha=0.5, color=colors[i])[0]
        tree_outlines.append(outline)
        tree_fills.append(fill)

    # Create bounding box rectangle
    bounding_rect = Rectangle(
        (0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2, linestyle="--"
    )
    axis.add_patch(bounding_rect)

    # Create text artist
    text = axis.text(
        0.05,
        0.95,
        "",
        transform=axis.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="white", ec="black"),
    )

    # Set up initial axis properties
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    return tree_outlines, tree_fills, bounding_rect, text
