import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Rectangle
from shapely.ops import unary_union
import numpy as np

from utils.tree import ChristmasTree
from utils.color_map import get_colors

from .bounding_square import calculate_bounding_square
from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)


@dataclass
class HighlightTreeData:
    has_collision: bool


@dataclass
class Snapshot:
    trees: list[ChristmasTree]
    side_length: Decimal
    selected_trees: dict[int, HighlightTreeData] = field(default_factory=dict)
    text: str = ""
    title: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


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

def update_artists_between_snapshots(
    axis,
    artists,
    snapshot: Snapshot,
    colors,
    prev_snapshot: Snapshot | None = None,
):
    tree_outlines, tree_fills, bounding_rect, text = artists
    max_trees = len(tree_outlines)
    trees = snapshot.trees
    scale_factor = trees[0]._scale_factor
    side_length = snapshot.side_length
    selected_trees = snapshot.selected_trees

    # Get previous frame data for change detection
    prev_trees = prev_snapshot.trees if prev_snapshot else []
    prev_selected_trees = prev_snapshot.selected_trees if prev_snapshot else {}
    prev_side_length = prev_snapshot.side_length if prev_snapshot else None

    artists_to_update = []

    def tree_changed(i, tree):
        """Check if tree position or selection state changed."""
        if prev_snapshot is None or i >= len(prev_trees):
            return True

        prev_tree = prev_trees[i]
        if (
            tree.center_x != prev_tree.center_x
            or tree.center_y != prev_tree.center_y
            or tree.angle != prev_tree.angle
        ):
            return True

        is_selected = i in selected_trees
        was_selected = i in prev_selected_trees
        if is_selected != was_selected:
            return True

        if is_selected and was_selected:
            if (
                selected_trees[i].has_collision
                != prev_selected_trees[i].has_collision
            ):
                return True

        return False

    # Update each tree (only if it changed)
    for i, tree in enumerate(trees):
        if not tree_changed(i, tree):
            continue
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [float(Decimal(str(val)) / scale_factor)
             for val in x_scaled]
        y = [float(Decimal(str(val)) / scale_factor)
             for val in y_scaled]

        # Check if this tree is highlighted
        is_highlighted = selected_trees and i in selected_trees
        has_collision = is_highlighted and selected_trees[i].has_collision

        # Update outline
        tree_outlines[i].set_data(x, y)
        if is_highlighted:
            tree_outlines[i].set_color("red")
            tree_outlines[i].set_linewidth(3)
        else:
            tree_outlines[i].set_color(colors[i])
            tree_outlines[i].set_linewidth(1)
        artists_to_update.append(tree_outlines[i])

        # Update fill
        tree_fills[i].set_xy(np.column_stack([x, y]))
        if has_collision:
            tree_fills[i].set_color("red")
            tree_fills[i].set_alpha(0.7)
        else:
            tree_fills[i].set_color(colors[i])
            tree_fills[i].set_alpha(0.5)
        artists_to_update.append(tree_fills[i])

    # Hide unused tree artists
    for i in range(len(trees), max_trees):
        tree_outlines[i].set_data([], [])
        tree_fills[i].set_xy(np.array([]).reshape(0, 2))

    # Update bounding box (only if side length changed)
    bounding_box_changed = prev_snapshot is None or side_length != prev_side_length
    if trees and bounding_box_changed:
        all_polygons = [t.polygon for t in trees]
        bounds = unary_union(all_polygons).bounds

        minx = Decimal(str(bounds[0])) / scale_factor
        miny = Decimal(str(bounds[1])) / scale_factor
        maxx = Decimal(str(bounds[2])) / scale_factor
        maxy = Decimal(str(bounds[3])) / scale_factor

        width = maxx - minx
        height = maxy - miny

        square_x = minx if width >= height else minx - \
            (side_length - width) / 2
        square_y = miny if height >= width else miny - \
            (side_length - height) / 2

        bounding_rect.set_xy((float(square_x), float(square_y)))
        bounding_rect.set_width(float(side_length))
        bounding_rect.set_height(float(side_length))

        # Update axis limits
        padding = Decimal(0.5)
        axis.set_xlim(minx - padding, maxx + padding)
        axis.set_ylim(miny - padding, maxy + padding)

    if bounding_box_changed:
        artists_to_update.append(bounding_rect)

    # Update text and title
    text.set_text(snapshot.text)
    axis.set_title(snapshot.title)
    artists_to_update.append(text)

    return artists_to_update

def plot_configuration(trees, side_length=None, title=None, show=True) -> plt.Axes:
    """
    Visualize a tree configuration with its bounding square.

    Args:
        trees: List of ChristmasTree objects
        side_length: Side length of bounding square (optional, will be calculated if not provided)
        title: Plot title
        show: Whether to call plt.show() immediately (default True)
    """
    _, axis = plt.subplots(figsize=(8, 8))
    artists = get_artists_for_configuration(
        axis,
        get_colors(len(trees)),
    )
    update_artists_between_snapshots(
        axis,
        artists,
        Snapshot(trees=trees, side_length=side_length or calculate_bounding_square(trees)),
        get_colors(len(trees)),
    )
    plt.tight_layout()
    if show:
        plt.show()
    return axis
