# %%
# Auto-reload modules before executing code - great for development!
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('.')  # Add parent directory to path
from utils.tree_utils import ChristmasTree

# %%
# Now you can use ChristmasTree here!
# Example:
tree = ChristmasTree(center_x='1.0', center_y='2.0', angle='45')
print(f"Tree created at ({tree.center_x}, {tree.center_y}) with angle {tree.angle}°")

