# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for Santa 2025 Challenge focused on optimizing Christmas tree packing. The goal is to pack N trees (from 1 to 200) into the smallest possible square bounding box, minimizing the sum of (bounding_box_area / N) across all configurations.

## Environment Setup

**Conda environment name:** `santa-challenge`

Create and activate the environment:
```bash
conda env create -f environment.yaml
conda activate santa-challenge
```

Key dependencies: Python 3.12, Jupyter, shapely 2.1.2, pandas, numpy, matplotlib

## Development Workflow

### Starting Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

### Jupytext Integration

This project uses jupytext to sync notebooks with Python files. The configuration in `jupytext.toml` automatically pairs `.ipynb` files with `.py:percent` files. When you save a notebook, the corresponding Python file is updated and vice versa.

**Important:** When modifying utility modules imported into notebooks, use the autoreload magic commands to see changes without restarting the kernel:
```python
%load_ext autoreload
%autoreload 2
```

### Scoring Submissions

Run the metric notebook or script to score a submission:
```bash
python santa-2025-metric.py
# or open santa-2025-metric.ipynb in Jupyter
```

The scoring function expects a CSV with columns: `id`, `x`, `y`, `deg` where values are prefixed with 's' (e.g., 's0.123').

## Code Architecture

### Core Components

**ChristmasTree Class** (`my_approaches/utils/tree.py`):
- Represents a rotatable Christmas tree with fixed dimensions
- Uses high-precision Decimal arithmetic to avoid floating-point errors
- Creates a shapely Polygon representing the tree geometry with trunk and three tiers
- **Internal Implementation:** Uses `_scale_factor = 1e18` (private class variable) to maintain numerical precision in Shapely operations. Matches `santa-2025-metric.py` for consistency with official scoring.
- **Public Interface:** Provides clean methods for tree manipulation:
  - `tree.move(dx, dy)` - translate tree by delta
  - `tree.rotate(angle_delta)` - rotate tree by delta angle
  - `tree.set_transform(x, y, angle)` - set absolute position and rotation
- **CRITICAL ARCHITECTURE RULE:** Approach code should NEVER import or use `_scale_factor` directly. Work only with logical coordinates (`center_x`, `center_y`, `angle`) and use tree methods. The scale factor is an implementation detail for Shapely precision.

**Tree Placement Algorithm** (`initialize_trees` in santa-2025-getting-started.py):
- Greedy approach that builds on previous N-tree configurations
- Places trees at weighted random angles (biased toward corners using `abs(sin(2*angle))`)
- Uses "approach from distance" strategy: starts tree at radius 20, moves inward until collision, backs up to find valid position
- Employs STRtree spatial index for efficient collision detection
- Makes 10 random attempts per tree and keeps the placement with minimum radius

**Collision Detection:**
- Uses shapely's STRtree for spatial indexing
- Trees can touch but not intersect: `polygon.intersects(other) and not polygon.touches(other)` indicates overlap
- Applied both during placement and metric validation

**Submission Format:**
- Index format: `{n:03d}_{tree_index}` (e.g., `001_0`, `002_0`, `002_1`, ...)
- Columns: `id`, `x`, `y`, `deg`
- All numeric values prefixed with 's' to preserve precision
- Values rounded to 6 decimal places
- Position limits: -100 to 100 for x and y coordinates

### Project Structure

- `santa-2025-getting-started.py/.ipynb`: Main starter code with tree placement algorithm and visualization (reference only, not used in custom approaches)
- `santa-2025-metric.py/.ipynb`: Official scoring metric implementation (DO NOT MODIFY)
- `sample_submission.csv`: Generated submission file
- `my_approaches/`: Custom solution approaches
  - `my_approaches/utils/tree.py`: Shared ChristmasTree class with methods for tree manipulation
  - `my_approaches/utils/collision.py`: Collision detection using STRtree spatial indexing
  - `my_approaches/utils/bounding_square.py`: Calculate bounding square for scoring
  - `my_approaches/utils/place_tree.py`: Greedy tree placement initialization
  - `my_approaches/utils/plot.py`: Visualization utilities
  - `my_approaches/0_simulated_annealing.py/.ipynb`: Simulated annealing optimization approach

### Key Constraints and Considerations

1. **Decimal Precision:** All calculations use Python's Decimal type with 25-digit precision to avoid floating-point errors in geometric calculations
2. **Scaling Factor (Internal):** Shapely polygons are internally scaled by 1e18 to maintain precision. This is hidden in the ChristmasTree class - approach code should never handle scaling directly.
3. **Bounding Square:** Score uses the largest dimension of the bounding rectangle to form a square (not just the rectangle area)
4. **Progressive Building:** Optimization approaches can reuse previous (N-1)-tree configurations when building N-tree configurations, which speeds up generation but may not find global optima

### Architectural Best Practices

**Separation of Concerns:**
- **Utility modules** (`my_approaches/utils/`): Handle internal implementation details like scale factors, polygon geometry, and Shapely operations
- **Approach modules** (`my_approaches/*.py`): Focus only on the optimization algorithm logic, working with logical coordinates
- **Rule:** Approach code should work in "logical space" - only manipulating tree positions (center_x, center_y, angle) and using utility functions/methods
- **Anti-pattern:** Importing `_scale_factor`, directly manipulating `polygon` attributes, or using `affinity.translate/rotate` in approach code

## Rules
- When modifying code that affects the veracity of this file, ensure to update this documentation accordingly.
