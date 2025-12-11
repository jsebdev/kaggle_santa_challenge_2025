# %%
%load_ext autoreload
%autoreload 2

import sys

from my_approaches.utils.plot import plot_configuration
from my_approaches.utils.tree import ChristmasTree
sys.path.append('..')

# %%
tree = ChristmasTree(center_x='0', center_y='0', angle='0')

plot_configuration(trees=[tree], scale_factor=1, title="Single Tree Test")

