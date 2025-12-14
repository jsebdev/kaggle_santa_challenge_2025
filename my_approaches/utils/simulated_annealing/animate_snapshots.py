from dataclasses import dataclass, field
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
from shapely.ops import unary_union
import logging

from utils.plot import HighlightTreeData
from utils.tree import ChristmasTree


logger = logging.getLogger(__name__)

@dataclass
class Snapshot:
    trees: list[ChristmasTree]
    side_length: Decimal
    selected_trees: dict[int, HighlightTreeData] = field(default_factory=dict)


def create_animation_from_snapshots(snapshots: list[Snapshot], fps=10):
    logger.debug('>>>>> animate_snapshots.py:26 "len(snapshots)"')
    logger.debug(len(snapshots))

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))

    # Initialize with first snapshot to set up artists
    first_snapshot = snapshots[0]
    first_trees = first_snapshot.trees
    max_trees = max(len(s.trees) for s in snapshots)

    # Get scale factor from first tree
    scale_factor = first_trees[0]._scale_factor if first_trees else Decimal(1)

    # Create color map
    colors = plt.cm.viridis([i / max(max_trees, 1) for i in range(max_trees)])

    # Create artist objects for each tree (outline and fill)
    tree_outlines = []
    tree_fills = []
    for i in range(max_trees):
        outline, = ax1.plot([], [], color=colors[i], linewidth=1)
        fill = ax1.fill([], [], alpha=0.5, color=colors[i])[0]
        tree_outlines.append(outline)
        tree_fills.append(fill)

    # Create bounding box rectangle
    bounding_rect = Rectangle((0, 0), 1, 1, fill=False, edgecolor='red',
                               linewidth=2, linestyle='--')
    ax1.add_patch(bounding_rect)


    # Create text artist
    text = ax1.text(
        0.05, 0.95,
        "",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle="round", fc="white", ec="black")
    )

    # Set up initial axis properties
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)

    def update(frame: int):
        logger.debug("0")
        snapshot = snapshots[frame]
        trees = snapshot.trees
        side_length = snapshot.side_length
        selected_trees = snapshot.selected_trees

        # Get previous frame data for change detection
        prev_snapshot = snapshots[frame - 1] if frame > 0 else None
        prev_trees = prev_snapshot.trees if prev_snapshot else []
        prev_selected_trees = prev_snapshot.selected_trees if prev_snapshot else {}
        prev_side_length = prev_snapshot.side_length if prev_snapshot else None

        artists_to_return = []

        def tree_changed(i, tree):
            """Check if tree position or selection state changed."""
            if frame == 0 or i >= len(prev_trees):
                return True

            prev_tree = prev_trees[i]
            if (tree.center_x != prev_tree.center_x or
                tree.center_y != prev_tree.center_y or
                tree.angle != prev_tree.angle):
                return True

            is_selected = i in selected_trees
            was_selected = i in prev_selected_trees
            if is_selected != was_selected:
                return True

            if is_selected and was_selected:
                if selected_trees[i].has_collision != prev_selected_trees[i].has_collision:
                    return True

            return False

        logger.debug("1")
        # Update each tree (only if it changed)
        for i, tree in enumerate(trees):
            if tree_changed(i, tree):
                x_scaled, y_scaled = tree.polygon.exterior.xy
                x = [float(Decimal(str(val)) / scale_factor) for val in x_scaled]
                y = [float(Decimal(str(val)) / scale_factor) for val in y_scaled]

                # Check if this tree is highlighted
                is_highlighted = selected_trees and i in selected_trees
                has_collision = is_highlighted and selected_trees[i].has_collision

                # Update outline
                tree_outlines[i].set_data(x, y)
                if is_highlighted:
                    tree_outlines[i].set_color('red')
                    tree_outlines[i].set_linewidth(3)
                else:
                    tree_outlines[i].set_color(colors[i])
                    tree_outlines[i].set_linewidth(1)
                artists_to_return.append(tree_outlines[i])

                # Update fill
                tree_fills[i].set_xy(np.column_stack([x, y]))
                if has_collision:
                    tree_fills[i].set_color('red')
                    tree_fills[i].set_alpha(0.7)
                else:
                    tree_fills[i].set_color(colors[i])
                    tree_fills[i].set_alpha(0.5)
                artists_to_return.append(tree_fills[i])
        logger.debug("2")

        # Hide unused tree artists
        for i in range(len(trees), max_trees):
            tree_outlines[i].set_data([], [])
            tree_fills[i].set_xy(np.array([]).reshape(0, 2))

        logger.debug("3")
        # Update bounding box (only if side length changed)
        bounding_box_changed = frame == 0 or side_length != prev_side_length
        if trees and bounding_box_changed:
            all_polygons = [t.polygon for t in trees]
            bounds = unary_union(all_polygons).bounds

            minx = Decimal(str(bounds[0])) / scale_factor
            miny = Decimal(str(bounds[1])) / scale_factor
            maxx = Decimal(str(bounds[2])) / scale_factor
            maxy = Decimal(str(bounds[3])) / scale_factor

            width = maxx - minx
            height = maxy - miny

            square_x = minx if width >= height else minx - (side_length - width) / 2
            square_y = miny if height >= width else miny - (side_length - height) / 2

            bounding_rect.set_xy((float(square_x), float(square_y)))
            bounding_rect.set_width(float(side_length))
            bounding_rect.set_height(float(side_length))

            # Update axis limits
            padding = Decimal(0.5)
            ax1.set_xlim(minx - padding, maxx + padding)
            ax1.set_ylim(miny - padding, maxy + padding)

        logger.debug("4")
        if bounding_box_changed:
            artists_to_return.append(bounding_rect)

        # Update text and title
        text.set_text(f'Iteration: {frame}')
        ax1.set_title(f'Tree Configuration\n{len(trees)} Trees: Side = {side_length:.6f}')
        artists_to_return.append(text)
        logger.debug("5")

        return artists_to_return

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000/fps,
                         repeat=True, blit=True)

    return anim
