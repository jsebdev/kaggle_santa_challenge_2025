from typing import List, Tuple
from decimal import Decimal
import math
from copy import deepcopy

from .collision import has_collision
from .bounding_square import calculate_bounding_square
from .tree import ChristmasTree
from .plot import Snapshot


class TreeState:
    """Represents the dynamic state of a tree during simulation."""

    def __init__(self, tree: ChristmasTree):
        self.tree = tree
        # Linear velocity (vx, vy)
        self.velocity = [0.0, 0.0]
        # Angular velocity (rad/s)
        self.angular_velocity = 0.0
        # Force accumulators
        self.force = [0.0, 0.0]
        self.torque = 0.0

    def reset_forces(self):
        """Reset force and torque accumulators."""
        self.force = [0.0, 0.0]
        self.torque = 0.0

    def add_force(self, fx: float, fy: float, px: float = None, py: float = None):
        """
        Add a force to this tree.

        Args:
            fx, fy: Force components
            px, py: Point of application (for torque calculation). If None, force applied at center.
        """
        self.force[0] += fx
        self.force[1] += fy

        # Calculate torque if application point is specified
        if px is not None and py is not None:
            cx, cy = float(self.tree.center_x), float(self.tree.center_y)
            # r = position vector from center to application point
            rx, ry = px - cx, py - cy
            # torque = r × F (cross product in 2D)
            self.torque += rx * fy - ry * fx


class ForceDirectedOptimizer:
    """
    Force-directed placement optimizer using physics simulation.

    The optimizer treats trees as rigid bodies and simulates forces:
    - Centripetal: spring force toward origin
    - Repulsion: point-to-point repulsion between nearby trees
    - Collision: strong penalty for overlapping trees
    """

    def __init__(
        self,
        k_center: float = 0.5,
        k_repulsion: float = 100.0,
        repulsion_radius: float = 10.0, # I'd make this smaller
        k_collision: float = 500.0,
        damping: float = 0.8,
        angular_damping: float = 0.7,
        mass: float = 1.0,
        moment_of_inertia: float = 1.0,
        dt: float = 0.01,
        max_velocity: float = 5.0,
        max_angular_velocity: float = 0.5,
    ):
        """
        Initialize optimizer with physics parameters.

        Args:
            k_center: Spring constant for center attraction
            k_repulsion: Repulsion strength coefficient
            repulsion_radius: Maximum distance for repulsion forces
            k_collision: Collision penalty coefficient (should be large)
            damping: Linear velocity damping (0-1, closer to 1 = more damping)
            angular_damping: Angular velocity damping (0-1)
            mass: Mass of each tree (for F=ma)
            moment_of_inertia: Rotational inertia (for τ=Iα)
            dt: Time step for simulation
            max_velocity: Maximum linear velocity (for stability)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.k_center = k_center
        self.k_repulsion = k_repulsion
        self.repulsion_radius = repulsion_radius
        self.k_collision = k_collision
        self.damping = damping
        self.angular_damping = angular_damping
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_angular_velocity = max_angular_velocity

    def sample_points_on_tree(self, tree: ChristmasTree, n_samples: int = 8) -> List[Tuple[float, float]]:
        """
        Sample points on the tree polygon for force calculations.

        Args:
            tree: ChristmasTree instance
            n_samples: Number of points to sample

        Returns:
            List of (x, y) coordinate tuples in logical coordinates
        """
        # Get polygon vertices in logical coordinates (unscaled)
        coords = tree.get_polygon_points()

        # Sample vertices and some midpoints
        points = []
        n_vertices = len(coords)

        # Add vertices
        for i in range(0, n_vertices, max(1, n_vertices // n_samples)):
            points.append(coords[i])

        # Add some midpoints for better coverage
        remaining = n_samples - len(points)
        if remaining > 0:
            step = max(1, n_vertices // remaining)
            for i in range(0, n_vertices, step):
                if len(points) >= n_samples:
                    break
                x1, y1 = coords[i]
                x2, y2 = coords[(i + 1) % n_vertices]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                points.append((mx, my))

        return points[:n_samples]

    def calculate_center_force(self, state: TreeState):
        """
        Calculate spring force pulling tree toward origin.

        Uses Hooke's law: F = -k * distance
        """
        cx, cy = float(state.tree.center_x), float(state.tree.center_y)
        distance = math.sqrt(cx**2 + cy**2)

        if distance > 1e-6:
            # Normalized direction toward center
            dx, dy = -cx / distance, -cy / distance
            # Spring force magnitude
            force_mag = self.k_center * distance
            # Apply force at center (no torque)
            state.add_force(force_mag * dx, force_mag * dy)

    def calculate_repulsion_forces(self, state: TreeState, other_states: List[TreeState]):
        """
        Calculate repulsion forces between this tree and others.

        Uses point-to-point repulsion with inverse square law.
        Automatically generates torque from off-center forces.
        """
        my_points = self.sample_points_on_tree(state.tree)

        for other_state in other_states:
            if other_state is state:
                continue

            other_points = self.sample_points_on_tree(other_state.tree)

            # Calculate pairwise repulsion between sampled points
            for px1, py1 in my_points:
                for px2, py2 in other_points:
                    dx, dy = px1 - px2, py1 - py2
                    dist_sq = dx**2 + dy**2
                    dist = math.sqrt(dist_sq)

                    # Only apply repulsion within radius
                    if dist < self.repulsion_radius and dist > 1e-6:
                        # Inverse square repulsion
                        force_mag = self.k_repulsion / dist_sq
                        # Normalized direction
                        fx = force_mag * dx / dist
                        fy = force_mag * dy / dist
                        # Apply force at point (generates torque)
                        state.add_force(fx, fy, px1, py1)

    def calculate_collision_forces(self, state: TreeState, other_states: List[TreeState]):
        """
        Calculate strong penalty forces for overlapping trees.

        When trees overlap, apply a strong separating force from the
        overlap centroid toward each tree's center.
        """
        for other_state in other_states:
            if other_state is state:
                continue

            # Check for collision using utility function
            if has_collision([state.tree, other_state.tree]):
                # Calculate separation direction (from other tree to this tree)
                cx1, cy1 = float(state.tree.center_x), float(state.tree.center_y)
                cx2, cy2 = float(other_state.tree.center_x), float(other_state.tree.center_y)

                dx, dy = cx1 - cx2, cy1 - cy2
                dist = math.sqrt(dx**2 + dy**2)

                if dist > 1e-6:
                    # Strong separation force
                    force_mag = self.k_collision
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    # Apply at center for stability
                    state.add_force(fx, fy)

    def apply_forces_and_update(self, states: List[TreeState]):
        """
        Update tree positions and rotations based on accumulated forces.

        Uses Euler integration with velocity damping for stability.
        """
        for state in states:
            # Linear dynamics: F = ma → a = F/m
            ax = state.force[0] / self.mass
            ay = state.force[1] / self.mass

            # Update velocity with damping
            state.velocity[0] = state.velocity[0] * (1 - self.damping) + ax * self.dt
            state.velocity[1] = state.velocity[1] * (1 - self.damping) + ay * self.dt

            # Clamp velocity for stability
            v_mag = math.sqrt(state.velocity[0]**2 + state.velocity[1]**2)
            if v_mag > self.max_velocity:
                scale = self.max_velocity / v_mag
                state.velocity[0] *= scale
                state.velocity[1] *= scale

            # Update position
            new_x = float(state.tree.center_x) + state.velocity[0] * self.dt
            new_y = float(state.tree.center_y) + state.velocity[1] * self.dt

            # Angular dynamics: τ = Iα → α = τ/I
            angular_accel = state.torque / self.moment_of_inertia

            # Update angular velocity with damping
            state.angular_velocity = state.angular_velocity * (1 - self.angular_damping) + angular_accel * self.dt

            # Clamp angular velocity
            if abs(state.angular_velocity) > self.max_angular_velocity:
                state.angular_velocity = math.copysign(self.max_angular_velocity, state.angular_velocity)

            # Update rotation (convert to degrees for tree API)
            new_angle = float(state.tree.angle) + math.degrees(state.angular_velocity * self.dt)

            # Apply transformation to tree
            state.tree.set_transform(
                Decimal(str(new_x)),
                Decimal(str(new_y)),
                Decimal(str(new_angle))
            )

            # Reset forces for next iteration
            state.reset_forces()

    def calculate_energy(self, states: List[TreeState]) -> float:
        """
        Calculate total system energy for monitoring convergence.

        Returns:
            Total kinetic + potential energy
        """
        kinetic = 0.0
        potential = 0.0

        for state in states:
            # Kinetic energy: (1/2)mv² + (1/2)Iω²
            v_sq = state.velocity[0]**2 + state.velocity[1]**2
            kinetic += 0.5 * self.mass * v_sq
            kinetic += 0.5 * self.moment_of_inertia * state.angular_velocity**2

            # Potential energy from center spring: (1/2)kx²
            cx, cy = float(state.tree.center_x), float(state.tree.center_y)
            dist_sq = cx**2 + cy**2
            potential += 0.5 * self.k_center * dist_sq

        return kinetic + potential

    def optimize(
        self,
        trees: List[ChristmasTree],
        n_iterations: int = 1000,
        convergence_threshold: float = 1e-4,
        verbose: bool = False,
        animate: bool = False,
        animation_interval: int = 50
    ):
        """
        Run force-directed optimization on a list of trees.

        Args:
            trees: List of ChristmasTree instances to optimize
            n_iterations: Maximum number of simulation steps
            convergence_threshold: Stop if energy change is below this threshold
            verbose: Print progress information
            animate: Whether to capture snapshots for animation
            animation_interval: Capture every N iterations (only if animate=True)

        Returns:
            If animate=False: List of optimized trees (same objects, modified in place)
            If animate=True: Dictionary with keys:
                - 'trees': Optimized list of trees
                - 'energy': Final energy value
                - 'snapshots': List of Snapshot objects for animation
                - 'converged': Whether convergence was reached
                - 'final_iteration': Final iteration number
        """
        # Create state objects for each tree
        trees = [deepcopy(t) for t in trees]
        states = [TreeState(tree) for tree in trees]

        prev_energy = float('inf')
        converged = False
        final_iteration = 0
        snapshots = []

        def capture_snapshot(energy: float, energy_change: float, iteration: int):
            """Helper to capture a snapshot."""
            bounding_square = calculate_bounding_square(trees)
            snapshot = Snapshot(
                trees=[deepcopy(t) for t in trees],
                side_length=bounding_square,
                text=f"Iteration: {iteration}\nEnergy: {energy:.6f}\nEnergy Change: {energy_change:.6f}",
                metrics={
                    # "iteration": iteration,
                    "energy": float(energy),
                    "energy_change": float(energy_change)
                }
            )
            snapshots.append(snapshot)

        # Capture initial snapshot if animating
        if animate:
            bounding_square = calculate_bounding_square(trees)
            capture_snapshot(prev_energy, 0.0, 0)

        for iteration in range(n_iterations):
            final_iteration = iteration

            # Calculate all forces
            for state in states:
                state.reset_forces()
                self.calculate_center_force(state)
                self.calculate_repulsion_forces(state, states)
                self.calculate_collision_forces(state, states)

            # Update positions and rotations
            self.apply_forces_and_update(states)

            # Check convergence and capture snapshots
            if iteration % animation_interval == 0 or iteration == n_iterations - 1:
                energy = self.calculate_energy(states)
                energy_change = abs(energy - prev_energy)

                if verbose:
                    print(f"Iteration {iteration}: Energy = {energy:.6f}, Change = {energy_change:.6f}")

                # Capture snapshot if animating
                if animate:
                    capture_snapshot(energy, energy_change, iteration)

                if energy_change < convergence_threshold:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    converged = True
                    break

                prev_energy = energy

        return {
            'trees': trees,
            'energy': prev_energy,
            'snapshots': snapshots,
            'converged': converged,
            'final_iteration': final_iteration
        }
