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


def get_artists_for_metrics(axis, metrics):
    """
    Initialize artists for metrics plot.

    Args:
        axis: Matplotlib axis for the metrics plot
        metrics: Dictionary of metric names to values (from first snapshot)

    Returns:
        Tuple of (line_artists_dict, text_artist)
    """
    metric_names = sorted(metrics.keys())  # Sort for consistency
    # Get enough colors for all metrics
    colors = get_colors(max(len(metric_names), 10))

    # Create line artists for each metric
    line_artists = {}
    for i, metric_name in enumerate(metric_names):
        line, = axis.plot([], [], label=metric_name, color=colors[i], linewidth=2, marker='o', markersize=2, alpha=0.8)
        line_artists[metric_name] = line

    # Create legend
    axis.legend(loc='best', fontsize=9, framealpha=0.9)

    # Create text artist for current values
    text = axis.text(
        0.02, 0.98, "",
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9)
    )

    # Set up axis
    axis.set_xlabel('Frame', fontsize=10)
    axis.set_ylabel('Value', fontsize=10)
    axis.set_title('Metrics Over Time', fontsize=12)
    axis.grid(True, alpha=0.3)

    return line_artists, text
