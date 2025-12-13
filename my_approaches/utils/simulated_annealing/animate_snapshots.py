from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from utils.plot import add_configuration_to_axis


def create_animation_from_snapshots(snapshots, save_path=None, fps=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract data for plotting
    iterations = [s['iteration'] for s in snapshots]
    energies = [s['energy'] for s in snapshots]

    def update(frame):
        snapshot = snapshots[frame]
        trees = snapshot['trees']
        energy = snapshot['energy']
        temp = snapshot['temperature']
        iteration = snapshot['iteration']

        # Clear axes
        ax1.clear()
        ax2.clear()

        # Plot tree configuration
        for tree in trees:
            poly = tree.polygon
            x, y = poly.exterior.xy
            ax1.fill(x, y, alpha=0.5, fc='green', ec='darkgreen', linewidth=1.5)

        side = energy
        ax1.add_patch(Rectangle((-side/2, -side/2), side, side,
                                fill=False, edgecolor='red', linewidth=2, linestyle='--'))

        ax1.set_xlim(-side/2 - 1, side/2 + 1)
        ax1.set_ylim(-side/2 - 1, side/2 + 1)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'Configuration - {len(trees)} Trees\nEnergy: {energy:.6f}', fontsize=12)

        # Plot energy progress
        current_iterations = iterations[:frame+1]
        current_energies = energies[:frame+1]

        ax2.plot(current_iterations, current_energies, color='green', linewidth=2)
        ax2.scatter([iteration], [energy], color='red', s=100, zorder=5)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Energy (Side Length)')
        ax2.set_title(f'Optimization Progress\nIteration: {iteration}, Temp: {temp:.4f}')
        ax2.grid(True, alpha=0.3)

        # Set consistent y-axis limits
        ax2.set_ylim(min(energies) * 0.95, max(energies) * 1.05)

        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000/fps, repeat=True)

    return anim

def create_animation_from_snapshots2(snapshots, save_path=None, fps=10):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract data for plotting
    iterations = [s['iteration'] for s in snapshots]
    energies = [s['energy'] for s in snapshots]

    def update(frame):
        snapshot = snapshots[frame]
        trees = snapshot['trees']
        energy = snapshot['energy']
        temp = snapshot['temperature']
        iteration = snapshot['iteration']

        # Clear axes
        ax1.clear()
        ax2.clear()

        add_configuration_to_axis(ax1, trees, side_length=energy)

        # Plot energy progress
        current_iterations = iterations[:frame+1]
        current_energies = energies[:frame+1]

        ax2.plot(current_iterations, current_energies, color='green', linewidth=2)
        ax2.scatter([iteration], [energy], color='red', s=100, zorder=5)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Energy (Side Length)')
        ax2.set_title(f'Optimization Progress\nIteration: {iteration}, Temp: {temp:.4f}')
        ax2.grid(True, alpha=0.3)

        # Set consistent y-axis limits
        # ax2.set_ylim(min(energies) * 0.95, max(energies) * 1.05)

        # plt.tight_layout()


    # def update(frame):
        # x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
        # y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
        # xp = x[:frame]
        # yp = y[:frame]
        # ax1.clear()
        # ax1.scatter(xp, yp)
        # return ax1

    # anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000/fps, repeat=True)
    anim = FuncAnimation(fig, update, frames=5, interval=1000/fps, repeat=True)

    return anim
