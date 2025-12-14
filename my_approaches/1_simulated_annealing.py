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

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from utils.simulated_annealing.animate_snapshots import Snapshot, create_animation_from_snapshots
from utils.tree import ChristmasTree
from utils.collision import has_collision
from utils.bounding_square import calculate_bounding_square
from utils.place_tree import initialize_greedy
from utils.plot import HighlightTreeData, plot_configuration
from utils.logging import configure_logging

# %%
configure_logging('1_simulated_annealing.log')
logger = logging.getLogger(__name__)
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

def caputure_animation_snapshots(snapshots: list[Snapshot],
                                 trees: list[ChristmasTree],
                                 energy,
                                 has_dollision=False,
                                 moved_tree_idxs=None):
    snapshot = Snapshot(
        trees=[deepcopy(t) for t in trees],
        side_length=energy,
    )
    if moved_tree_idxs is not None:
        for idx in moved_tree_idxs:
            snapshot.selected_trees[idx] = HighlightTreeData(has_collision=has_dollision)
    snapshots.append(snapshot)


def simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_rate=0.995,
    iterations_per_temp=100,
    verbose=True,
    animate=False,
    animation_interval=1,
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
    best_iteration = 0

    temperature = initial_temp
    history = []
    snapshots = []  # Store configurations for animation
    total_iterations = 0
    accepted_moves = 0
    temp_step = 0

    n_trees = len(current_trees)

    if animate:
        caputure_animation_snapshots(snapshots, current_trees, current_energy,)

    # Main optimization loop: continue until temperature is very low
    while temperature > final_temp:
        # At each temperature, try many random perturbations
        for _ in range(iterations_per_temp):
            total_iterations += 1
            # logger.debug('>>>>> 1_simulated_annealing.py:210 "total_iterations"')
            # logger.debug(total_iterations)

            # Randomly choose a type of perturbation
            move_type = random.choice(['translate', 'rotate'])

            # Generate a new candidate configuration
            tree_idx = random.randint(0, n_trees - 1)
            moved_tree_idxs = [tree_idx]
            if move_type == 'translate':
                new_trees = perturb_translation(current_trees, tree_idx, max_delta=0.1)
                # new_trees = perturb_translation(current_trees, tree_idx, max_delta=1)
            # elif move_type == 'rotate':
            else:  # swap
                new_trees = perturb_rotation(current_trees, tree_idx, max_angle=20)

            # Reject configurations with overlapping trees
            collistion = has_collision(new_trees)
            # logger.debug('>>>>> 1_simulated_annealing.py:246 "moved_tree_idxs"')
            # logger.debug(moved_tree_idxs)
            # logger.debug('>>>>> 1_simulated_annealing.py:246 "move_type"')
            # logger.debug(move_type)
            # logger.debug('>>>>> 1_simulated_annealing.py:248 "collistion"')
            # logger.debug(collistion)
            if animate and (total_iterations % animation_interval == 0):
                caputure_animation_snapshots(snapshots, new_trees, best_energy,
                                             has_dollision=collistion, moved_tree_idxs=moved_tree_idxs)
            if collistion:
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
                    best_iteration = total_iterations

        history.append({
            'temperature': temperature,
            'current_energy': float(current_energy),
            'best_energy': best_energy,
            'acceptance_rate': accepted_moves / total_iterations
        })


        if verbose and len(history) % 10 == 0:
            print(f"Temp: {temperature:.4f}, Best: {best_energy:.6f}, "
                  f"Current: {current_energy:.6f}, Accept: {accepted_moves}/{total_iterations}")

        # Cool down: temperature *= cooling_rate (e.g., 0.995)
        # This gradually reduces temperature, making the algorithm more greedy over time
        temperature *= cooling_rate
        temp_step += 1

    print("Simulated annealing complete.")
    print("total iterations:", total_iterations)
    print("total temperature steps:", temp_step)
    print("accepted moves:", accepted_moves)
    print("acceptance rate:", accepted_moves / total_iterations)

    return {
        'best_trees': best_trees,
        'best_energy': best_energy,
        'history': history,
        'snapshots': snapshots,
        'best_iteration': best_iteration,
    }


# %%
n_trees = 3
seed = 42
random.seed(seed)
initial_trees = initialize_greedy(n_trees)

result = simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.98,
    cooling_rate=0.98,
    iterations_per_temp=50,
    verbose=True,
    animate=True,
    animation_interval=1,
)
# best_trees, best_energy, history, snapshots
best_trees = result['best_trees']
best_energy = result['best_energy']
snapshots = result['snapshots']
best_iteration = result['best_iteration']

print(f"Best configuration found at iteration {best_iteration} with bounding square side length: {best_energy}")

# plot_configuration(best_trees, side_length=best_energy)

anim = create_animation_from_snapshots(snapshots, fps=10)
from IPython.display import HTML
plt.close()
HTML(anim.to_jshtml())  # Display animation as HTML5 video
