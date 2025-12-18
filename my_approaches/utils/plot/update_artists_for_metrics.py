import logging


logger = logging.getLogger(__name__)


def update_artists_for_metrics(axis, artists, precomputed_metric_data, snapshots, frame):
    line_artists, text = artists
    artists_to_update = []

    # Slice pre-computed data up to current frame
    current_metric_data = {}
    for metric_name, data in precomputed_metric_data.items():
        current_metric_data[metric_name] = {
            'x': data['x'][:frame + 1],
            'y': data['y'][:frame + 1]
        }

    # Update line artists
    for metric_name, data in current_metric_data.items():
        if metric_name in line_artists:
            line_artists[metric_name].set_data(data['x'], data['y'])
            artists_to_update.append(line_artists[metric_name])

    # Update axis limits - filter out inf values
    if current_metric_data:
        all_y_values = []
        for data in current_metric_data.values():
            # Filter out inf and -inf values
            finite_values = [y for y in data['y'] if y != float('inf') and y != float('-inf')]
            all_y_values.extend(finite_values)

        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max - y_min
            if y_range > 0:
                padding = y_range * 0.1
                axis.set_ylim(y_min - padding, y_max + padding)
            else:
                # All values are the same, add some padding
                axis.set_ylim(y_min - 1, y_max + 1)
        axis.set_xlim(-0.5, max(frame + 2, 10))

    # Update text with current values
    current_metrics = snapshots[frame].metrics
    text_lines = [f"Frame: {frame}"]
    text_lines.append("")  # Empty line for spacing
    for metric_name in sorted(current_metrics.keys()):
        value = current_metrics[metric_name]
        if value == float('inf'):
            text_lines.append(f"{metric_name}: inf")
        elif value == float('-inf'):
            text_lines.append(f"{metric_name}: -inf")
        else:
            text_lines.append(f"{metric_name}: {value:.6f}")
    text.set_text('\n'.join(text_lines))
    artists_to_update.append(text)

    return artists_to_update
