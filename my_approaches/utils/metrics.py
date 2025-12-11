from decimal import Decimal, getcontext
from shapely.ops import unary_union
from shapely.strtree import STRtree


class ParticipantVisibleError(Exception):
    pass


def score_group(placed_trees):
    scale_factor = placed_trees[0]._scale_factor
    num_trees = len(placed_trees)

    # Check for collisions using neighborhood search
    all_polygons = [p.polygon for p in placed_trees]
    r_tree = STRtree(all_polygons)

    # Checking for collisions
    for i, poly in enumerate(all_polygons):
        indices = r_tree.query(poly)
        for index in indices:
            if index == i:  # don't check against self
                continue
            if poly.intersects(all_polygons[index]) and not poly.touches(all_polygons[index]):
                raise ParticipantVisibleError(f'Overlapping trees in group')

    # Calculate score for the group
    bounds = unary_union(all_polygons).bounds
    # Use the largest edge of the bounding rectangle to make a square boulding box
    side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

    group_score = (Decimal(side_length_scaled) ** 2) / (scale_factor**2) / Decimal(num_trees)
    print(f'Group score: {group_score}')
