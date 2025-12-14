import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.patches import Rectangle
from shapely.ops import unary_union
import numpy as np

from utils.tree import ChristmasTree
from utils.color_map import get_colors

from utils.bounding_square import calculate_bounding_square
from dataclasses import dataclass, field
import logging



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
