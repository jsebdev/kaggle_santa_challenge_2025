import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.plot import Snapshot, get_artists_for_configuration, update_artists_between_snapshots
from utils.color_map import get_colors


logger = logging.getLogger(__name__)


def create_animation_from_snapshots(snapshots: list[Snapshot], fps=10):
    fig, axis = plt.subplots(1, 1, figsize=(16, 7))

    # Initialize with first snapshot to set up artists
    max_trees = max(len(s.trees) for s in snapshots)
    colors = get_colors(max_trees)
    artists = get_artists_for_configuration(
        axis,
        colors=colors,
    )

    def update(frame: int):
        snapshot = snapshots[frame]
        prev_snapshot = snapshots[frame - 1] if frame > 0 else None
        return update_artists_between_snapshots(
            axis,
            artists,
            snapshot,
            colors,
            prev_snapshot,
        )

    anim = FuncAnimation(
        fig, update, frames=len(snapshots), interval=1000 / fps, repeat=True, blit=True
    )

    return anim
