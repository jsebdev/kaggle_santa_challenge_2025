# %%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('..')

from utils.plot import plot_configuration
from utils.tree import ChristmasTree

# %%
trees = [
    ChristmasTree(center_x=0, center_y=0, angle=0),
    ChristmasTree(center_x=1, center_y=0, angle=0),
]

plot_configuration(trees, side_length=1)
