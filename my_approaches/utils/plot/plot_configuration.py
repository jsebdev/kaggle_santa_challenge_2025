import matplotlib.pyplot as plt

from utils.color_map import get_colors_for_trees

from utils.plot import get_artists_for_configuration, update_artists_between_snapshots
from utils.bounding_square import calculate_bounding_square
from utils.plot.models import Snapshot



def plot_configuration(trees, side_length=None, show=True) -> plt.Axes:
    _, axis = plt.subplots(figsize=(8, 8))
    artists = get_artists_for_configuration(
        axis,
        get_colors_for_trees(len(trees)),
    )
    update_artists_between_snapshots(
        axis,
        artists,
        Snapshot(trees=trees, side_length=side_length or calculate_bounding_square(trees)),
        get_colors_for_trees(len(trees)),
    )
    plt.tight_layout()
    if show:
        plt.show()
    return axis
