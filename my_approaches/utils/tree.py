"""Shared utilities for the Santa 2025 Kaggle Challenge.

This module provides the ChristmasTree class with methods for tree manipulation.

IMPORTANT: ChristmasTree._scale_factor is an internal implementation detail for
maintaining numerical precision in Shapely geometric operations. Approach code
should NEVER use _scale_factor directly. Use the tree's methods instead.
"""

from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon

# Set precision for Decimal
getcontext().prec = 25

# Fixed tree dimensions
trunk_w = Decimal('0.15')
trunk_h = Decimal('0.2')
base_w = Decimal('0.7')
mid_w = Decimal('0.4')
top_w = Decimal('0.25')
tip_y = Decimal('0.8')
tier_1_y = Decimal('0.5')
tier_2_y = Decimal('0.25')
base_y = Decimal('0.0')
trunk_bottom_y = -trunk_h
tree_points = [
    # Start at Tip
    (Decimal("0.0"), tip_y),
    # Right side - Top Tier
    (top_w / Decimal("2"), tier_1_y),
    (top_w / Decimal("4"), tier_1_y),
    # Right side - Middle Tier
    (mid_w / Decimal("2"), tier_2_y),
    (mid_w / Decimal("4"), tier_2_y),
    # Right side - Bottom Tier
    (base_w / Decimal("2"), base_y),
    # Right Trunk
    (trunk_w / Decimal("2"), base_y),
    (
        trunk_w / Decimal("2"),
        trunk_bottom_y,
    ),
    # Left Trunk
    (
        -(trunk_w / Decimal("2")),
        trunk_bottom_y,
    ),
    (
        -(trunk_w / Decimal("2")),
        base_y,
    ),
    # Left side - Bottom Tier
    (
        -(base_w / Decimal("2")),
        base_y,
    ),
    # Left side - Middle Tier
    (
        -(mid_w / Decimal("4")),
        tier_2_y,
    ),
    (
        -(mid_w / Decimal("2")),
        tier_2_y,
    ),
    # Left side - Top Tier
    (
        -(top_w / Decimal("4")),
        tier_1_y,
    ),
    (
        -(top_w / Decimal("2")),
        tier_1_y,
    ),
]

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size.

    The tree has fixed dimensions and uses high-precision Decimal arithmetic
    internally. Approach code should work with center_x, center_y, and angle
    attributes in logical coordinates, and use the methods to manipulate trees.
    """

    # Internal scaling factor to maintain precision in Shapely operations.
    # Matches santa-2025-metric.py (1e18) to ensure consistency with official scoring.
    # This is private - utility functions can access it, but approach code should not.
    _scale_factor: Decimal = Decimal('1e18')

    def __init__(
        self,
        center_x: str | float = "0",
        center_y: str | float = "0",
        angle: str | float = "0",
    ):
        """Initializes the Christmas tree with a specific position and rotation.

        Args:
            center_x: X coordinate of tree center (in logical space)
            center_y: Y coordinate of tree center (in logical space)
            angle: Rotation angle in degrees
        """
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        # Build the polygon geometry
        self.polygon = self._build_polygon()

    def _build_polygon(self):
        """Build the tree polygon at the current position and rotation.

        This is an internal method that creates the Shapely polygon geometry
        in scaled space for numerical precision.

        Returns:
            Shapely Polygon object
        """
        # Build polygon in scaled space for Shapely precision
        scaled_points = [(x * self._scale_factor, y * self._scale_factor) for x, y in tree_points]
        initial_polygon = Polygon(scaled_points)
        rotated = affinity.rotate(
            initial_polygon, float(self.angle), origin=(0, 0))
        return affinity.translate(rotated,
                                  xoff=float(self.center_x *
                                             self._scale_factor),
                                  yoff=float(self.center_y * self._scale_factor))

    def move(self, dx, dy):
        """Move this tree by a delta in x and y directions.

        Updates both the logical position (center_x, center_y) and the internal
        polygon geometry. Works in-place.

        Args:
            dx: Change in x coordinate (Decimal or convertible to Decimal)
            dy: Change in y coordinate (Decimal or convertible to Decimal)
        """
        dx = Decimal(str(dx))
        dy = Decimal(str(dy))

        self.center_x += dx
        self.center_y += dy
        # Rebuild polygon from scratch to avoid floating-point error accumulation
        self.polygon = self._build_polygon()

    def rotate(self, angle_delta):
        """Rotate this tree by a delta angle.

        Updates both the logical angle and rebuilds the internal polygon geometry.
        Works in-place.

        Args:
            angle_delta: Change in rotation angle in degrees (Decimal or convertible)
        """
        angle_delta = Decimal(str(angle_delta))
        self.angle += angle_delta
        self.polygon = self._build_polygon()

    def set_transform(self, x, y, angle):
        """Set the absolute position and rotation of this tree.

        Updates both the logical coordinates and rebuilds the internal polygon.
        Works in-place.

        Args:
            x: New x coordinate (Decimal or convertible to Decimal)
            y: New y coordinate (Decimal or convertible to Decimal)
            angle: New rotation angle in degrees (Decimal or convertible)
        """
        self.center_x = Decimal(str(x))
        self.center_y = Decimal(str(y))
        self.angle = Decimal(str(angle))
        self.polygon = self._build_polygon()

    def get_polygon_points(self):
        """Get the polygon points in logical (unscaled) coordinates.

        Returns the actual world coordinates of the tree polygon vertices,
        properly unscaled from the internal representation.

        Returns:
            List of (x, y) tuples representing polygon vertices in logical space
        """
        # Get the scaled polygon coordinates
        coords = list(self.polygon.exterior.coords)
        # Unscale and return as list of tuples
        return [(float(Decimal(x) / self._scale_factor), float(Decimal(y) / self._scale_factor)) for x, y in coords]
