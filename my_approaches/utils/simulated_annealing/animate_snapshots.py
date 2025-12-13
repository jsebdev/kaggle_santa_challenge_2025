from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import add_configuration_to_axis
import logging


logger = logging.getLogger(__name__)


def create_animation_from_snapshots(snapshots, save_path=None, fps=10):
    print('>>>>> animate_snapshots.py:13 "len(snapshots)"')
    print(len(snapshots))

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))

    # Extract data for plotting
    iterations = [s['iteration'] for s in snapshots]
    energies = [s['energy'] for s in snapshots]

    def update(frame):
        snapshot = snapshots[frame]
        trees = snapshot['trees']
        energy = snapshot['energy']
        iteration = snapshot['iteration']
        logger.debug('>>>>> animate_snapshots.py:27 "iteration"')
        logger.debug(iteration)
        selected_trees = snapshot.get('selected_trees', None)
        logger.debug('>>>>> animate_snapshots.py:29 "selected_trees"')
        logger.debug(selected_trees)

        # Clear axes
        ax1.clear()

        add_configuration_to_axis(ax1, trees, side_length=energy, highlighted_trees=selected_trees)

        # Add iteration label
        ax1.set_title(f'Iteration: {iteration}')

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000/fps, repeat=True)

    return anim
