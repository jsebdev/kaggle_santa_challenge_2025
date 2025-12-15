from copy import deepcopy
import numpy as np
import random
from decimal import Decimal

from shapely import affinity
from shapely.strtree import STRtree

from .tree import ChristmasTree
from utils.plot.models import Snapshot


def initialize_at_corner(num_trees, seed=None, capture_snapshots=False):
    if seed is not None:
        random.seed(seed)

    placed_trees = []
    snapshots: list[Snapshot] = []
    for i in range(num_trees):
        theta = random.uniform(0,np.pi/2)
        vx = Decimal(str(np.cos(theta)))  # x component of unit vector
        vy = Decimal(str(np.sin(theta)))  # y component of unit vector
        radius = Decimal('10.0')
        tree = ChristmasTree(
            center_x=radius * vx,
            center_y=radius * vy,
        )
        placed_trees.append(tree)
        if capture_snapshots:
            snapshots.append(Snapshot(
                trees=[deepcopy(t) for t in placed_trees],
            ))

    return placed_trees, snapshots
