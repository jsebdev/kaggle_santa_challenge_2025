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
from decimal import Decimal
from copy import deepcopy

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from utils.animate import create_animation_from_snapshots
from utils.tree import ChristmasTree
from utils.collision import has_collision
from utils.bounding_square import calculate_bounding_square
from utils.initialize_greedy_0 import initialize_greedy_0
from utils.plot import HighlightTreeData, Snapshot, plot_configuration
from utils.logging import configure_logging

# %%
configure_logging('1_simulated_annealing.log')
logger = logging.getLogger(__name__)
pd.set_option('display.float_format', '{:.12f}'.format)

# %%
def calculate_energy(trees):
    return calculate_bounding_square(trees)


# %%
def perturb_translation(trees, tree_idx, max_delta=0.1):
    new_trees = [deepcopy(t) for t in trees]
    tree = new_trees[tree_idx]

    dx = random.uniform(-max_delta, max_delta)
    dy = random.uniform(-max_delta, max_delta)

    tree.move(dx, dy)

    return new_trees


def perturb_rotation(trees, tree_idx, max_angle=15):
    new_trees = [deepcopy(t) for t in trees]
    tree = new_trees[tree_idx]

    delta_angle = random.uniform(-max_angle, max_angle)

    tree.rotate(delta_angle)

    return new_trees

# %%
def capture_animation_snapshots(snapshots: list[Snapshot],
                                trees: list[ChristmasTree],
                                energy: Decimal,
                                iteration: int,
                                temperature: float,
                                accepted_moves: int,
                                has_dollision=False,
                                moved_tree_idxs=None,
                                ):
    snapshot = Snapshot(
        trees=[deepcopy(t) for t in trees],
        side_length=energy,
            text=f"Iteration: {iteration}\nBounding Square Side Length: {energy:.6f}\naccepted moves: {accepted_moves}",
        metrics={
            "temperature": temperature,
            "side_length": float(energy),
        }
    )
    if moved_tree_idxs is not None:
        for idx in moved_tree_idxs:
            snapshot.selected_trees[idx] = HighlightTreeData(has_collision=has_dollision)
    snapshots.append(snapshot)

# %%
def simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_rate=0.995,
    iterations_per_temp=100,
    verbose=False,
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
    best_accepted_move = 0

    temperature = initial_temp
    history = []
    snapshots = []  # Store configurations for animation
    total_iterations = 0
    accepted_moves = 0
    temp_step = 0

    n_trees = len(current_trees)

    if animate:
        capture_animation_snapshots(snapshots, current_trees, current_energy, total_iterations, temperature, accepted_moves)

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
            if collistion:
                if animate and (total_iterations % animation_interval == 0):
                    capture_animation_snapshots(snapshots, new_trees, current_energy, total_iterations, temperature, accepted_moves,
                                                 has_dollision=collistion, moved_tree_idxs=moved_tree_idxs)
                continue

            # Calculate how much worse/better the new configuration is
            new_energy = calculate_energy(new_trees)
            delta_energy = float(new_energy - current_energy)

            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_trees = new_trees
                current_energy = new_energy
                accepted_moves += 1
                if animate and (total_iterations % animation_interval == 0):
                    capture_animation_snapshots(snapshots, new_trees, current_energy, total_iterations, temperature, accepted_moves,
                                                 has_dollision=collistion, moved_tree_idxs=moved_tree_idxs)

                # Track the best solution ever seen
                if current_energy < best_energy:
                    best_trees = [deepcopy(t) for t in current_trees]
                    best_energy = current_energy
                    best_iteration = total_iterations
                    best_accepted_move = accepted_moves

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
        'best_accepted_move': best_accepted_move,
        'total_iterations': total_iterations,
        'total_accepted_moves': accepted_moves,
    }


# %%
n_trees = 3
seed = 42
random.seed(seed)
initial_trees = initialize_greedy_0(n_trees)

result = simulated_annealing(
    initial_trees,
    initial_temp=1.0,
    final_temp=0.01,
    cooling_rate=0.98,
    iterations_per_temp=100,
    verbose=False,
    animate=True,
    animation_interval=10,
)
# %%
best_trees = result['best_trees']
best_energy = result['best_energy']
snapshots = result['snapshots']
best_iteration = result['best_iteration']
best_accepted_move = result['best_accepted_move']
total_accepted_moves = result['total_accepted_moves']
total_iterations = result['total_iterations']
print(f"Best configuration found at move {best_accepted_move} with bounding square side length: {best_energy}")
print(f"Total iterations: {total_iterations}, Total accepted moves: {total_accepted_moves}")
print(f"total number of snapshots captured: {len(snapshots)}")
# %%
plot_configuration(best_trees, side_length=best_energy)

# %%
anim = create_animation_from_snapshots(snapshots, fps=10, metrics_factor={
    "side_length": 1.0,
    "temperature": 1.0,
})
# from IPython.display import HTML
# plt.close()
# HTML(anim.to_jshtml())  # Display animation as HTML5 video

