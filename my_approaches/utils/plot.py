import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from utils.tree import ChristmasTree

from .bounding_square import calculate_bounding_square
from dataclasses import dataclass
import logging


@dataclass
class HighlightTreeData:
    has_collision: bool


logger = logging.getLogger(__name__)


def add_configuration_to_axis(
    ax: plt.Axes,
    trees: list[ChristmasTree],
    side_length: Decimal | None = None,
    highlighted_trees: dict[int, HighlightTreeData] | None = None,
    title: str | None = None,
) -> plt.Axes:
    num_trees = len(trees)
    colors = plt.cm.viridis([i / max(num_trees, 1) for i in range(num_trees)])

    if not trees:
        return ax

    all_polygons = [t.polygon for t in trees]
    # unary_union merges all tree polygons into one big shape
    # .bounds gives us (minx, miny, maxx, maxy) of that combined shape
    bounds = unary_union(all_polygons).bounds

    scale_factor = trees[0]._scale_factor

    for i, tree in enumerate(trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(str(val)) / scale_factor for val in x_scaled]
        y = [Decimal(str(val)) / scale_factor for val in y_scaled]
        if highlighted_trees and i in highlighted_trees:
            ax.plot(x, y, color='red', linewidth=3)
            if highlighted_trees[i].has_collision:
                ax.fill(x, y, alpha=0.7, color='red')
        else:
            ax.plot(x, y, color=colors[i], linewidth=1)
            ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width = maxx - minx
    height = maxy - miny

    if side_length is None:
        side_length = calculate_bounding_square(trees)

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2

    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(bounding_square)

    padding = 0.5
    ax.set_xlim(float(minx - Decimal(str(padding))),
                float(maxx + Decimal(str(padding))))
    ax.set_ylim(float(miny - Decimal(str(padding))),
                float(maxy + Decimal(str(padding))))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    if not title:
        title = "Tree Configuration"
    ax.set_title(f'{title}\n{len(trees)} Trees: Side = {side_length:.6f}')
    return ax


def plot_configuration(trees, side_length=None, title=None, show=True) -> plt.Axes:
    """
    Visualize a tree configuration with its bounding square.

    Args:
        trees: List of ChristmasTree objects
        side_length: Side length of bounding square (optional, will be calculated if not provided)
        title: Plot title
        show: Whether to call plt.show() immediately (default True)
    """
    _, ax = plt.subplots(figsize=(8, 8))
    ax = add_configuration_to_axis(ax, trees, side_length, highlighted_trees=None, title=title)
    plt.tight_layout()
    if show:
        plt.show()
    return ax
