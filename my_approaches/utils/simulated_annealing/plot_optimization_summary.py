def plot_optimization_summary(history, snapshots=None):
    """
    Create a comprehensive visualization of the optimization process.

    Args:
        history: History dictionary from simulated_annealing
        snapshots: Optional snapshots for showing configuration evolution
    """
    if snapshots and len(snapshots) > 0:
        # Create multi-panel figure
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Energy over time
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = list(range(len(history)))
        best_energies = [h['best_energy'] for h in history]
        current_energies = [h['current_energy'] for h in history]

        ax1.plot(iterations, best_energies, label='Best Energy', color='green', linewidth=2)
        ax1.plot(iterations, current_energies, label='Current Energy', color='blue', alpha=0.5)
        ax1.set_xlabel('Temperature Step')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Temperature decay
        ax2 = fig.add_subplot(gs[0, 2:])
        temperatures = [h['temperature'] for h in history]
        ax2.semilogy(iterations, temperatures, color='red', linewidth=2)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Temperature (log scale)')
        ax2.set_title('Temperature Decay')
        ax2.grid(True, alpha=0.3)

        # Show configurations at different stages
        n_snapshots_to_show = min(4, len(snapshots))
        snapshot_indices = [int(i * (len(snapshots)-1) / (n_snapshots_to_show-1))
                           for i in range(n_snapshots_to_show)]

        for i, snap_idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(gs[1, i])
            snapshot = snapshots[snap_idx]
            trees = snapshot['trees']
            energy = snapshot['energy']

            for tree in trees:
                poly = tree.polygon
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='green', ec='darkgreen', linewidth=1)

            side = energy
            ax.add_patch(Rectangle((-side/2, -side/2), side, side,
                                  fill=False, edgecolor='red', linewidth=1.5, linestyle='--'))

            ax.set_xlim(-side/2 - 0.5, side/2 + 0.5)
            ax.set_ylim(-side/2 - 0.5, side/2 + 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Step {snapshot["iteration"]}\nE={energy:.4f}', fontsize=10)
            ax.grid(True, alpha=0.2)

        plt.suptitle('Optimization Summary', fontsize=16, fontweight='bold')
        plt.show()
    else:
        # Simple plot without snapshots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = list(range(len(history)))
        best_energies = [h['best_energy'] for h in history]
        current_energies = [h['current_energy'] for h in history]

        ax1.plot(iterations, best_energies, label='Best Energy', color='green', linewidth=2)
        ax1.plot(iterations, current_energies, label='Current Energy', color='blue', alpha=0.5)
        ax1.set_xlabel('Temperature Step')
        ax1.set_ylabel('Energy')
        ax1.set_title('Energy vs Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        temperatures = [h['temperature'] for h in history]
        ax2.semilogy(iterations, temperatures, color='red', linewidth=2)
        ax2.set_xlabel('Temperature Step')
        ax2.set_ylabel('Temperature (log scale)')
        ax2.set_title('Temperature Decay')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

