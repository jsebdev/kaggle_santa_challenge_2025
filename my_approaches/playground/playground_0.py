# %%
%load_ext autoreload
%autoreload 2

import sys
import matplotlib.pyplot as plt

sys.path.append('..')
sys.path.append('../../')

from my_approaches.utils.animate import create_animation_from_snapshots
from my_approaches.utils.initialize_at_corner import initialize_at_corner
from utils.plot import plot_configuration
from utils.tree import ChristmasTree


# %%
trees, snapshots = initialize_at_corner(num_trees=1, seed=42, capture_snapshots=True)

# %%
anim = create_animation_from_snapshots(snapshots)
plt.close()
from IPython.display import HTML
HTML(anim.to_jshtml())  # Display animation as HTML5 video
