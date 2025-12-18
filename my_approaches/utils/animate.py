import logging
import traceback

from shapely import snap

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.plot import Snapshot, get_artists_for_configuration, get_artists_for_metrics, update_artists_between_snapshots, update_artists_for_metrics
from utils.color_map import get_colors_for_trees


logger = logging.getLogger(__name__)


def _precompute_metric_data(snapshots: list[Snapshot], metrics_factor: dict) -> dict:
    """Pre-compute all metric data once to avoid O(N²) complexity during animation."""
    metric_data = {}

    for i, snapshot in enumerate(snapshots):
        for metric_name, value in snapshot.metrics.items():
            if metric_name not in metric_data:
                metric_data[metric_name] = {'x': [], 'y': []}
            metric_data[metric_name]['x'].append(i)
            metric_factor = metrics_factor.get(metric_name, 1.0)
            metric_data[metric_name]['y'].append(value * metric_factor)

    return metric_data


def create_animation_from_snapshots(snapshots: list[Snapshot], metrics_factor = {}, fps=10):
    fig, (axis0, axis1) = plt.subplots(1, 2, figsize=(18, 9))

    # Initialize with first snapshot to set up artists
    snapshots[0].trees
    max_trees = max(len(s.trees) for s in snapshots)
    colors = get_colors_for_trees(max_trees)
    trees_artists = get_artists_for_configuration(
        axis0,
        colors=colors,
    )
    metrics_artists = get_artists_for_metrics(
        axis1,
        snapshots[0].metrics,
        metrics_factor,
    )

    # Pre-compute all metric data once (O(N) instead of O(N²))
    precomputed_metric_data = _precompute_metric_data(snapshots, metrics_factor)

    def update(frame: int):
        try:
            snapshot = snapshots[frame]
            prev_snapshot = snapshots[frame - 1] if frame > 0 else None
            artists_to_update_in_trees = update_artists_between_snapshots(
                axis0,
                trees_artists,
                snapshot,
                colors,
                prev_snapshot,
            )
            artists_to_update_in_metrics = update_artists_for_metrics(
                axis1,
                metrics_artists,
                precomputed_metric_data,
                snapshots,
                frame,
            )
            return artists_to_update_in_trees + artists_to_update_in_metrics
        except Exception as e:
            logger.error(f"Error updating frame {frame}: {e}")
            logger.error(traceback.format_exc())

    anim = FuncAnimation(
        fig, update, frames=len(snapshots), interval=1000 / fps, repeat=True, blit=True
    )

    return anim
