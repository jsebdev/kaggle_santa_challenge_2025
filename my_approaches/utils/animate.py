import logging

from shapely import snap

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.plot import Snapshot, get_artists_for_configuration, get_artists_for_metrics, update_artists_between_snapshots, update_artists_for_metrics
from utils.color_map import get_colors


logger = logging.getLogger(__name__)


def create_animation_from_snapshots(snapshots: list[Snapshot], fps=10):
    fig, (axis0, axis1) = plt.subplots(2, 1, figsize=(16, 7))
    # fig, (axis0) = plt.subplots(1, 1, figsize=(16, 7))

    # Initialize with first snapshot to set up artists
    max_trees = max(len(s.trees) for s in snapshots)
    colors = get_colors(max_trees)
    trees_artists = get_artists_for_configuration(
        axis0,
        colors=colors,
    )
    metrics_artists = get_artists_for_metrics(
        axis1,
        snapshots[0].metrics,
    )

    def update(frame: int):
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
            snapshots,
            frame,
        )
        return artists_to_update_in_trees + artists_to_update_in_metrics

    anim = FuncAnimation(
        fig, update, frames=len(snapshots), interval=1000 / fps, repeat=True, blit=True
    )

    return anim
