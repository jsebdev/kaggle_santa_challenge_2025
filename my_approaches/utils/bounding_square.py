from shapely.ops import unary_union
from decimal import Decimal


def calculate_bounding_square(trees) -> Decimal:
    """
    Calculate the side length of the minimum bounding square for a set of trees.

    The competition scores based on SQUARE area, not rectangle area.
    So we take max(width, height) to get the square's side length.

    Args:
        trees: List of ChristmasTree objects

    Returns:
        Decimal: Side length of the bounding square
    """
    if not trees:
        return Decimal('0')

    all_polygons = [t.polygon for t in trees]

    # unary_union merges all polygons into one combined shape
    # This is useful because .bounds on the union gives us the overall bounding box
    # bounds is a tuple: (minx, miny, maxx, maxy)
    bounds = unary_union(all_polygons).bounds

    # Convert from scaled coordinates back to original coordinates
    # Tree polygons are scaled up by _scale_factor for precision, so we divide here
    scale_factor = trees[0]._scale_factor
    minx = Decimal(str(bounds[0])) / scale_factor
    miny = Decimal(str(bounds[1])) / scale_factor
    maxx = Decimal(str(bounds[2])) / scale_factor
    maxy = Decimal(str(bounds[3])) / scale_factor

    width = maxx - minx
    height = maxy - miny

    # Return the larger dimension to form a SQUARE (not a rectangle)
    # The score is (side_length)² / n, so minimizing side_length is key
    return max(width, height)
