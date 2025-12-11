# %%
%load_ext autoreload
%autoreload 2

import sys

sys.path.append('..')
sys.path.append('../../')

from utils.plot import plot_configuration
from utils.tree import ChristmasTree


# %%
tree = ChristmasTree(center_x='0', center_y='0', angle='0')

plot_configuration(trees=[tree], title="Single Tree Test")

