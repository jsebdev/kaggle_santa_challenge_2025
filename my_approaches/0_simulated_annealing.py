# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Simulated Annealing Approach
#
# This notebook implements a simulated annealing algorithm to optimize tree packing.
# The algorithm iteratively perturbs tree positions and orientations, accepting
# improvements and occasionally accepting worse solutions to escape local minima.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
%matplotlib ipympl
import sys

sys.path.append('.')

# %%
import math
import random
from decimal import Decimal, getcontext
from copy import deepcopy
from matplotlib.animation import FuncAnimation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from utils.tree import ChristmasTree
from utils.collision import has_collision
from utils.bounding_square import calculate_bounding_square
from utils.place_tree import initialize_greedy
from utils.plot import plot_configuration

getcontext().prec = 25
pd.set_option('display.float_format', '{:.12f}'.format)

# %%
def calculate_energy(trees):
    """
    Calculate the energy (objective function) for the current configuration.
    Lower energy is better. In simulated annealing, "energy" is what we minimize.

    Args:
        trees: List of ChristmasTree objects

    Returns:
        Decimal: The bounding square side length (energy to minimize)
    """
    return calculate_bounding_square(trees)


# %%
def perturb_translation(trees, tree_idx, max_delta=0.1):
    """
    Create a new configuration by translating a single tree.
    This is one of three "move types" in our simulated annealing.

    Args:
        trees: Current list of ChristmasTree objects
        tree_idx: Index of tree to perturb
        max_delta: Maximum translation distance

    Returns:
        List of ChristmasTree objects (new configuration)
    """
    # Deep copy to avoid modifying original configuration
    new_trees = [deepcopy(t) for t in trees]
    tree = new_trees[tree_idx]

    # Random translation in both x and y directions
    dx = random.uniform(-max_delta, max_delta)
    dy = random.uniform(-max_delta, max_delta)

    # Move the tree using its clean method
    tree.move(dx, dy)

    return new_trees


def perturb_rotation(trees, tree_idx, max_angle=15):
    """
    Create a new configuration by rotating a single tree.
    Rotation can help trees fit together better in tight spaces.

    Args:
        trees: Current list of ChristmasTree objects
        tree_idx: Index of tree to perturb
        max_angle: Maximum rotation angle in degrees

    Returns:
        List of ChristmasTree objects (new configuration)
    """
    new_trees = [deepcopy(t) for t in trees]
    tree = new_trees[tree_idx]

    # Rotate by a random angle (positive or negative)
    delta_angle = random.uniform(-max_angle, max_angle)

    # Rotate the tree using its clean method
    tree.rotate(delta_angle)

    return new_trees


def perturb_swap(trees, idx1, idx2):
    """
    Create a new configuration by swapping positions of two trees.
    This can make larger changes to the configuration than translation/rotation.

    Args:
        trees: Current list of ChristmasTree objects
        idx1: Index of first tree
        idx2: Index of second tree

    Returns:
        List of ChristmasTree objects (new configuration)
    """
    new_trees = [deepcopy(t) for t in trees]

    # Save the positions of both trees
    pos1 = (new_trees[idx1].center_x, new_trees[idx1].center_y)
    pos2 = (new_trees[idx2].center_x, new_trees[idx2].center_y)

    # Swap positions (keeping rotations)
    new_trees[idx1].set_transform(pos2[0], pos2[1], new_trees[idx1].angle)
    new_trees[idx2].set_transform(pos1[0], pos1[1], new_trees[idx2].angle)

    return new_trees


# %%
def simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_rate=0.995,
    iterations_per_temp=100,
    verbose=True,
    animate=False,
    animation_interval=1
):
    """
    Optimize tree configuration using simulated annealing.

    Args:
        initial_trees: Starting configuration
        initial_temp: Starting temperature for annealing
        final_temp: Final temperature (stopping criterion)
        cooling_rate: Rate at which temperature decreases (0 < rate < 1)
        iterations_per_temp: Number of iterations at each temperature
        verbose: Whether to print progress
        animate: Whether to capture snapshots for animation
        animation_interval: Capture every N temperature steps (only if animate=True)

    Returns:
        tuple: (best_trees, best_energy, history, snapshots)
            snapshots is only populated if animate=True
    """
    current_trees = [deepcopy(t) for t in initial_trees]
    current_energy = calculate_energy(current_trees)

    best_trees = [deepcopy(t) for t in current_trees]
    best_energy = current_energy

    temperature = initial_temp
    history = []
    snapshots = []  # Store configurations for animation
    total_iterations = 0
    accepted_moves = 0
    temp_step = 0

    n_trees = len(current_trees)

    # Main optimization loop: continue until temperature is very low
    while temperature > final_temp:
        # At each temperature, try many random perturbations
        for _ in range(iterations_per_temp):
            total_iterations += 1

            # Randomly choose a type of perturbation
            move_type = random.choice(['translate', 'rotate', 'swap'])

            # Generate a new candidate configuration
            if move_type == 'translate':
                tree_idx = random.randint(0, n_trees - 1)
                new_trees = perturb_translation(current_trees, tree_idx, max_delta=0.15)
            elif move_type == 'rotate':
                tree_idx = random.randint(0, n_trees - 1)
                new_trees = perturb_rotation(current_trees, tree_idx, max_angle=20)
            else:  # swap
                if n_trees < 2:
                    continue
                idx1, idx2 = random.sample(range(n_trees), 2)
                new_trees = perturb_swap(current_trees, idx1, idx2)

            # Reject configurations with overlapping trees
            if has_collision(new_trees):
                continue

            # Calculate how much worse/better the new configuration is
            new_energy = calculate_energy(new_trees)
            delta_energy = float(new_energy - current_energy)

            # SIMULATED ANNEALING ACCEPTANCE CRITERION:
            # - Always accept improvements (delta_energy < 0)
            # - Sometimes accept worse solutions with probability exp(-delta_energy / temperature)
            #   * High temperature: accept many worse solutions (explore widely)
            #   * Low temperature: rarely accept worse solutions (exploit good regions)
            # This allows escaping local minima!
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_trees = new_trees
                current_energy = new_energy
                accepted_moves += 1

                # Track the best solution ever seen
                if current_energy < best_energy:
                    best_trees = [deepcopy(t) for t in current_trees]
                    best_energy = current_energy

        history.append({
            'temperature': temperature,
            'current_energy': float(current_energy),
            'best_energy': float(best_energy),
            'acceptance_rate': accepted_moves / total_iterations
        })

        # Capture snapshot for animation
        if animate and temp_step % animation_interval == 0:
            snapshots.append({
                'trees': [deepcopy(t) for t in best_trees],
                'energy': float(best_energy),
                'temperature': temperature,
                'iteration': temp_step
            })

        if verbose and len(history) % 10 == 0:
            print(f"Temp: {temperature:.4f}, Best: {best_energy:.6f}, "
                  f"Current: {current_energy:.6f}, Accept: {accepted_moves}/{total_iterations}")

        # Cool down: temperature *= cooling_rate (e.g., 0.995)
        # This gradually reduces temperature, making the algorithm more greedy over time
        temperature *= cooling_rate
        temp_step += 1

    return best_trees, best_energy, history, snapshots


# %%
def create_animation_from_snapshots(snapshots, save_path=None, fps=10):
    """
    Create an animation from captured snapshots using matplotlib.animation.
    This creates a smooth animation that can be displayed in Jupyter or saved as MP4/GIF.

    Args:
        snapshots: List of snapshot dictionaries from simulated_annealing
        save_path: Optional path to save animation (e.g., 'optimization.mp4' or 'optimization.gif')
        fps: Frames per second for the animation

    Returns:
        matplotlib.animation.FuncAnimation object
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract data for plotting
    iterations = [s['iteration'] for s in snapshots]
    energies = [s['energy'] for s in snapshots]
    temperatures = [s['temperature'] for s in snapshots]

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

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved!")

    return anim


# %%
def plot_optimization_summary(history, snapshots=None):
    """
    Create a comprehensive visualization of the optimization process.

    Args:
        history: History dictionary from simulated_annealing
        snapshots: Optional snapshots for showing configuration evolution
    """
    if snapshots and len(snapshots) > 0:
        # Create multi-panel figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Energy over time
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = list(range(len(history)))
        best_energies = [h['best_energy'] for h in history]
        current_energies = [h['current_energy'] for h in history]

        ax1.plot(iterations, best_energies, label='Best Energy', color='green', linewidth=2)
        ax1.plot(iterations, current_energies, label='Current Energy', color='blue', alpha=0.5)
        ax1.set_xlabel('Temperature Step')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Temperature decay
        ax2 = fig.add_subplot(gs[0, 2:])
        temperatures = [h['temperature'] for h in history]
        ax2.semilogy(iterations, temperatures, color='red', linewidth=2)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Temperature (log scale)')
        ax2.set_title('Temperature Decay')
        ax2.grid(True, alpha=0.3)

        # Show configurations at different stages
        n_snapshots_to_show = min(4, len(snapshots))
        snapshot_indices = [int(i * (len(snapshots)-1) / (n_snapshots_to_show-1))
                           for i in range(n_snapshots_to_show)]

        for i, snap_idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(gs[1, i])
            snapshot = snapshots[snap_idx]
            trees = snapshot['trees']
            energy = snapshot['energy']

            for tree in trees:
                poly = tree.polygon
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='green', ec='darkgreen', linewidth=1)

            side = energy
            ax.add_patch(Rectangle((-side/2, -side/2), side, side,
                                  fill=False, edgecolor='red', linewidth=1.5, linestyle='--'))

            ax.set_xlim(-side/2 - 0.5, side/2 + 0.5)
            ax.set_ylim(-side/2 - 0.5, side/2 + 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Step {snapshot["iteration"]}\nE={energy:.4f}', fontsize=10)
            ax.grid(True, alpha=0.2)

        plt.suptitle('Optimization Summary', fontsize=16, fontweight='bold')
        plt.show()
    else:
        # Simple plot without snapshots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = list(range(len(history)))
        best_energies = [h['best_energy'] for h in history]
        current_energies = [h['current_energy'] for h in history]

        ax1.plot(iterations, best_energies, label='Best Energy', color='green', linewidth=2)
        ax1.plot(iterations, current_energies, label='Current Energy', color='blue', alpha=0.5)
        ax1.set_xlabel('Temperature Step')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        temperatures = [h['temperature'] for h in history]
        ax2.semilogy(iterations, temperatures, color='red', linewidth=2)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Temperature (log scale)')
        ax2.set_title('Temperature Decay')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# %%
def optimize_all_configurations(
    max_trees=200,
    sa_params=None,
    visualize_every=20,
    seed=42
):
    """
    Optimize configurations for all tree counts from 1 to max_trees.

    Args:
        max_trees: Maximum number of trees to optimize
        sa_params: Dictionary of simulated annealing parameters
        visualize_every: Visualize every N configurations
        seed: Random seed for reproducibility

    Returns:
        dict: Mapping from tree count to (trees, side_length)
    """
    if sa_params is None:
        sa_params = {
            'initial_temp': 1.0,
            'final_temp': 0.01,
            'cooling_rate': 0.995,
            'iterations_per_temp': 100
        }

    random.seed(seed)
    configurations = {}

    for n in range(1, max_trees + 1):
        print(f"\n{'='*60}")
        print(f"Optimizing configuration for {n} trees...")
        print(f"{'='*60}")

        # PROGRESSIVE BUILDING STRATEGY:
        # Use the optimized (n-1)-tree configuration and add one more tree
        # This is faster than starting from scratch, but may not find global optimum
        if n == 1:
            initial_trees = initialize_greedy(n, seed=seed)
        else:
            # Start with previous solution + one new tree at origin
            prev_trees = configurations[n-1][0]
            new_tree = ChristmasTree(
                center_x='0',
                center_y='0',
                angle=str(random.uniform(0, 360))
            )
            initial_trees = [deepcopy(t) for t in prev_trees] + [new_tree]

        initial_energy = calculate_energy(initial_trees)
        print(f"Initial energy: {initial_energy:.6f}")
        # plot_configuration(initial_trees, title=f"Initial Configuration - {n} Trees")

        best_trees, best_energy, history, _ = simulated_annealing(
            initial_trees,
            verbose=(n % 10 == 0),
            animate=False,  # Don't capture snapshots in batch optimization
            **sa_params
        )

        print(f"Final energy: {best_energy:.6f}")
        print(f"Improvement: {float(initial_energy - best_energy):.6f} "
              f"({100 * (1 - float(best_energy/initial_energy)):.2f}%)")

        configurations[n] = (best_trees, best_energy)

        if n % visualize_every == 0 or n <= 5:
            plot_configuration(best_trees, side_length=best_energy,
                             title=f"Optimized Configuration - {n} Trees")

    return configurations


# %%
def configurations_to_submission(configurations):
    """
    Convert optimized configurations to Kaggle submission format.

    Args:
        configurations: Dict mapping tree count to (trees, side_length)

    Returns:
        pandas.DataFrame: Submission dataframe
    """
    index = [f'{n:03d}_{t}' for n in range(1, 201) for t in range(n)]
    tree_data = []

    for n in range(1, 201):
        trees, _ = configurations[n]
        for tree in trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])

    cols = ['x', 'y', 'deg']
    submission = pd.DataFrame(index=index, columns=cols, data=tree_data).rename_axis('id')

    for col in cols:
        submission[col] = submission[col].astype(float).round(decimals=6)

    for col in submission.columns:
        submission[col] = 's' + submission[col].astype('string')

    return submission


# %% [markdown]
# ## Run Optimization
#
# This cell runs the full optimization. Adjust parameters as needed.

# %%
# Example: Optimize a small subset for testing
test_configurations = optimize_all_configurations(
    max_trees=1,
    sa_params={
        'initial_temp': 1.0,
        'final_temp': 0.01,
        'cooling_rate': 0.98,
        'iterations_per_temp': 50
    },
    visualize_every=5,
    seed=42
)

# %%
# Calculate total score
total_score = sum(float(side**2) / n for n, (_, side) in test_configurations.items())
print(f"\nTotal score for {len(test_configurations)} configurations: {total_score:.6f}")


# %% [markdown]
# ## Animation Examples
#
# This section shows three different ways to animate the optimization process:
# 1. **Live animation** - Real-time updates in Jupyter (simplest, best for quick feedback)
# 2. **Snapshot-based animation** - Create smooth animations from captured states (best for presentations)
# 3. **Summary visualization** - Static overview of the entire optimization process

# %%
# Example 2: Snapshot-based Animation
# First, run optimization with snapshot capture
n_trees = 5
initial_trees = initialize_greedy(n_trees, seed=42)

best_trees, best_energy, history, snapshots = simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_rate=0.98,
    iterations_per_temp=50,
    verbose=True,
    animate=True,
    animation_interval=2  # Capture every 2 temperature steps
)

print(f"Captured {len(snapshots)} snapshots")
print(f"Final energy: {best_energy:.6f}")

# %%
# Create and display the animation
# Note: In Jupyter, this will display as an interactive animation
anim = create_animation_from_snapshots(snapshots, fps=5)

# To save the animation (uncomment one of these):
# anim = create_animation_from_snapshots(snapshots, save_path='optimization.gif', fps=5)
# anim = create_animation_from_snapshots(snapshots, save_path='optimization.mp4', fps=10)

# Display the animation in Jupyter
# from IPython.display import HTML
# HTML(anim.to_jshtml())

# %%
# Example 3: Optimization Summary
# Create a comprehensive static visualization
plot_optimization_summary(history, snapshots)

# %% [markdown]
# ## Full Run (Uncomment to execute)
#
# This will take a long time to complete. Consider running overnight or
# adjusting parameters for faster (but potentially lower quality) results.

# %%
# full_configurations = optimize_all_configurations(
#     max_trees=200,
#     sa_params={
#         'initial_temp': 2.0,
#         'final_temp': 0.001,
#         'cooling_rate': 0.997,
#         'iterations_per_temp': 200
#     },
#     visualize_every=20,
#     seed=42
# )
#
# submission = configurations_to_submission(full_configurations)
# submission.to_csv('submission_simulated_annealing.csv')
# print("\nSubmission saved to submission_simulated_annealing.csv")

# %%
