from utils.color_map import get_color_for_metrics

def get_artists_for_metrics(axis, metrics):
    metric_names = sorted(metrics.keys())  # Sort for consistency
    # Get enough colors for all metrics
    colors = get_color_for_metrics(len(metric_names))

    # Create line artists for each metric
    line_artists = {}
    for i, metric_name in enumerate(metric_names):
        line, = axis.plot([], [], label=metric_name, color=colors[i], linewidth=2, marker='o', markersize=2, alpha=0.8)
        line_artists[metric_name] = line

    # Create legend
    axis.legend(loc='best', fontsize=9, framealpha=0.9)

    # Create text artist for current values
    text = axis.text(
        0.02, 0.98, "",
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.9)
    )

    # Set up axis
    axis.set_xlabel('Frame', fontsize=10)
    axis.set_ylabel('Value', fontsize=10)
    axis.set_title('Metrics Over Time', fontsize=12)
    axis.grid(True, alpha=0.3)

    return line_artists, text
