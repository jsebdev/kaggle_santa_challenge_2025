def update_artists_for_metrics(axis, artists, snapshots, frame, metrics_factor):
    line_artists, text = artists
    artists_to_update = []

    # Collect all metric data up to current frame
    metric_data = {}

    for i in range(frame + 1):
        snapshot = snapshots[i]
        for metric_name, value in snapshot.metrics.items():
            if metric_name not in metric_data:
                metric_data[metric_name] = {'x': [], 'y': []}
            metric_data[metric_name]['x'].append(i)
            metric_factor = metrics_factor.get(metric_name, 1.0)
            metric_data[metric_name]['y'].append(value * metric_factor)

    # Update line artists
    for metric_name, data in metric_data.items():
        if metric_name in line_artists:
            line_artists[metric_name].set_data(data['x'], data['y'])
            artists_to_update.append(line_artists[metric_name])

    # Update axis limits
    if metric_data:
        all_y_values = []
        for data in metric_data.values():
            all_y_values.extend(data['y'])

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
        text_lines.append(f"{metric_name}: {value:.6f}")
    text.set_text('\n'.join(text_lines))
    artists_to_update.append(text)

    return artists_to_update
