import math
import random
from decimal import Decimal

from shapely import affinity
from shapely.strtree import STRtree

from my_approaches.utils.tree import ChristmasTree


def initialize_greedy(num_trees, scale_factor, seed=None):
    """
    Create an initial configuration using a simple greedy placement strategy.

    Args:
        num_trees: Number of trees to place
        seed: Random seed for reproducibility

    Returns:
        List of ChristmasTree objects
    """
    if seed is not None:
        random.seed(seed)

    if num_trees == 0:
        return []

    placed_trees = []

    for i in range(num_trees):
        angle = Decimal(str(random.uniform(0, 360)))

        if i == 0:
            placed_trees.append(ChristmasTree(center_x='0', center_y='0', angle=str(angle)))
            continue

        # Create a new tree (initially at origin, we'll move it later)
        tree = ChristmasTree(angle=str(angle))
        placed_polygons = [p.polygon for p in placed_trees]

        # STRtree is a spatial index for fast collision detection
        # Instead of checking all N trees (O(N)), STRtree.query() only returns nearby trees
        tree_index = STRtree(placed_polygons)

        best_pos = None
        min_dist = Decimal('Infinity')

        # Try 5 random approach directions and keep the best one
        for _ in range(5):
            # Pick a random direction vector (angle)
            theta = random.uniform(0, 2 * math.pi)
            vx = Decimal(str(math.cos(theta)))  # x component of unit vector
            vy = Decimal(str(math.sin(theta)))  # y component of unit vector

            # Start far from center (radius=10) and approach in steps
            radius = Decimal('10.0')
            step = Decimal('0.3')

            # Move inward until we hit a collision
            while radius >= 0:
                px = radius * vx  # Position = radius * direction
                py = radius * vy

                # Create a test polygon at this position
                candidate_poly = affinity.translate(
                    tree.polygon,
                    xoff=float(px * scale_factor),
                    yoff=float(py * scale_factor))

                # KEY: tree_index.query() returns INDICES of polygons whose bounding boxes
                # overlap with candidate_poly. This is MUCH faster than checking all trees!
                # Example: if we have 100 trees, this might only return [23, 45, 67]
                # (the 3 nearby trees) instead of checking all 100
                possible_indices = tree_index.query(candidate_poly)

                # Check only the nearby trees for actual collision
                # intersects() = shapes overlap, touches() = shapes share edge without overlap
                if any(
                    (
                        candidate_poly.intersects(placed_polygons[i]) and
                        not candidate_poly.touches(placed_polygons[i])
                    ) for i in possible_indices):
                    break  # Found collision, stop moving inward

                radius -= step

            # Back up a bit to ensure no collision (we stepped into collision zone)
            radius += step
            if radius < min_dist:
                min_dist = radius
                best_pos = (radius * vx, radius * vy)

        px, py = best_pos
        tree.center_x = px
        tree.center_y = py
        tree.polygon = affinity.translate(
            tree.polygon,
            xoff=float(px * scale_factor),
            yoff=float(py * scale_factor))

        placed_trees.append(tree)

    return placed_trees
