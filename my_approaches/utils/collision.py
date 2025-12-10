from shapely.strtree import STRtree


def has_collision(trees):
    """
    Check if any trees in the configuration overlap (not just touch).

    Uses STRtree spatial indexing for efficient collision detection.
    Without STRtree, we'd need O(n²) checks. With STRtree, we only check nearby trees.

    Args:
        trees: List of ChristmasTree objects

    Returns:
        bool: True if any trees overlap, False otherwise
    """
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]

    # STRtree = "Sort-Tile-Recursive" tree, a spatial index structure
    # It organizes polygons spatially so we can quickly find nearby objects
    tree_index = STRtree(polygons)

    for i, poly in enumerate(polygons):
        # tree_index.query(poly) returns INDICES of polygons whose bounding boxes
        # overlap with poly's bounding box. This is a FAST spatial filter!
        # Example: If we have 100 trees, query might only return [45, 67, 89]
        # (the nearby trees) instead of all 100 indices
        possible_collisions = tree_index.query(poly)

        for j in possible_collisions:
            if i != j:  # Don't check a polygon against itself
                # poly.intersects(other) = True if they overlap OR touch
                # poly.touches(other) = True if they share a boundary but don't overlap
                # We want to reject overlaps but allow touching, so:
                # collision = intersects AND NOT touches
                if poly.intersects(polygons[j]) and not poly.touches(polygons[j]):
                    return True
    return False
