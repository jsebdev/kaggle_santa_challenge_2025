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

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib ipympl
import sys
import matplotlib.pyplot as plt

sys.path.append('.')

import random
from utils.initialize_greedy_0 import initialize_greedy_0
from utils.plot.plot_configuration import plot_configuration
from utils.force_directed import ForceDirectedOptimizer
from utils.animate import create_animation_from_snapshots
from utils.logging import configure_logging

configure_logging('2_force_directed_placement.log')

# %%
n_trees = 30
seed = 42
random.seed(seed)
initial_trees = initialize_greedy_0(n_trees)

plot_configuration(initial_trees)

force_directed_optimizer = ForceDirectedOptimizer(
    k_center=1000,
    k_repulsion=2,
)
result = force_directed_optimizer.optimize(
    trees=initial_trees,
    animate=False,
    # animation_interval=100,
)
final_trees = result['trees']
snapshots = result['snapshots']

plot_configuration(final_trees)

# anim = create_animation_from_snapshots(snapshots, fps=10, metrics_factor={'energy_change': 100})
# from IPython.display import HTML
# plt.close()
# HTML(anim.to_jshtml())  # Display animation as HTML5 video
