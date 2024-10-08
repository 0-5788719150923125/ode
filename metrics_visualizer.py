import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from functools import reduce
import re
import numpy as np
from matplotlib.ticker import ScalarFormatter, AutoLocator
from textwrap import fill

def plot_metric(data, metric_key, metric_name, label_metrics):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    one_million_bytes = 1_000_000
    all_values = []
    labels = []  # Store label information for overlap detection
    max_step = 0  # Keep track of the maximum step across all runs

    for run in data:
        steps, metric_values = process_run_data(run, metric_key)
        total_params = run['totalParams']
        total_megabytes = total_params / one_million_bytes

        label_parts = [f"v{run['version']} | {total_megabytes:.2f}M | {run['runId']}", run['class']]
        for label_metric in label_metrics:
            value = get_nested_value(run, label_metric)
            if value:
                if isinstance(label_metric, str):
                    label_parts.append(f"{label_metric}: {str(value)}")
                else:
                    label_parts.append(f"{'.'.join(label_metric)}: {str(value)}")

        label = "\n".join(label_parts)
        line, = plt.plot(steps, metric_values, marker='o', label=fill(label, 40))  # Wrap the label text
        all_values.extend([v for v in metric_values if not np.isnan(v)])

        if len(steps) > 0:
            max_step = max(max_step, max(steps))  # Update the maximum step

            # Add annotation for the most recent (last) data point
            last_step = steps[-1]
            last_value = metric_values[-1]
            if not np.isnan(last_value):
                labels.append((last_step, last_value, f'{last_value:.4f}', label))

    # Calculate the x-shift based on the maximum step
    x_shift = 15.0

    # Sort labels by y-value (last_value) to handle overlaps from bottom to top
    labels.sort(key=lambda x: x[1])

    # Function to check if two labels overlap
    def labels_overlap(label1, label2, y_threshold=0.1):
        _, y1, _, _ = label1
        _, y2, _, _ = label2
        return abs(y1 - y2) < y_threshold

    # Adjust label positions to avoid overlaps
    adjusted_labels = []
    for label in labels:
        x, y, text, run_label = label
        new_y = y
        while any(labels_overlap((x, new_y, text, run_label), adj_label) for adj_label in adjusted_labels):
            new_y += 0.1  # Adjust this value to control vertical spacing
        adjusted_labels.append((x, new_y, text, run_label))

    # Plot adjusted labels
    for label_x, label_y, label_text, run_label in adjusted_labels:
        plt.annotate(label_text, 
                     xy=(label_x, label_y),
                     xytext=(x_shift, 0), textcoords='offset points',
                     ha='left', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1),
                     color='black',
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', lw=0.5))

    formatted_name = split_variable(metric_name)
    plt.title(f"{formatted_name} Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(formatted_name, fontsize=12)

    # Create legend with original settings
    plt.legend(title="ODE Runs", title_fontsize=12, loc='upper right', bbox_to_anchor=(1, 1),
               frameon=True, fancybox=True, shadow=True)

    # Calculate reasonable y-axis limits with margins
    y_min, y_max = calculate_ylim(all_values)

    # Use symlog scale
    linthresh = max(abs(y_min), abs(y_max)) * 1e-3  # Adjust this value as needed
    plt.yscale('symlog', linthresh=linthresh)

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    # Set up y-axis ticks and labels
    locator = AutoLocator()
    ax = plt.gca()
    ax.yaxis.set_major_locator(locator)

    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)

    # Ensure y-axis labels are visible
    plt.tick_params(axis='y', which='both', labelsize=10)

    # Ensure x-axis (steps) are always positive
    plt.xlim(left=0)

    plt.tight_layout()
    plt.savefig(f'metrics_{metric_key}.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_ylim(data, percentile_range=(5, 95), expansion_factor=1.2, margin_factor=0.05):
    """Calculate y-axis limits based on percentiles of the data with added margins."""
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return 0, 1  # Default range if no valid data

    lower, upper = np.percentile(data, percentile_range)
    data_min, data_max = np.min(data), np.max(data)

    # Calculate the range and add expansion
    data_range = upper - lower
    expanded_range = data_range * expansion_factor

    # Calculate initial y_min and y_max
    y_min = max(lower - (expanded_range - data_range) / 2, data_min)
    y_max = min(upper + (expanded_range - data_range) / 2, data_max)

    # Add margins
    margin = (y_max - y_min) * margin_factor
    y_min -= margin
    y_max += margin

    # Ensure positive values for log scale
    if y_min <= 0:
        y_min = min(data[data > 0]) / 10  # Use the smallest positive value / 10

    return y_min, y_max

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_variable(string):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', string)
    return ' '.join(word.capitalize() for word in words)

def process_run_data(run, metric_key):
    validate_every = run['configuration']['validateEvery']
    current_step = run['step']
    metric_values = run[metric_key][::-1]

    # Clean the data: remove None values and convert to float
    metric_values = [float(v) if v is not None else np.nan for v in metric_values]

    num_steps = len(metric_values)
    
    # Calculate the last validation step
    last_validation_step = current_step - (current_step % validate_every)
    
    # Generate steps backwards from the last validation step
    steps = [last_validation_step - (validate_every * i) for i in range(num_steps)][::-1]

    return steps, metric_values

def get_nested_value(data, keys):
    if isinstance(keys, str):
        keys = keys.split('.')
    return reduce(lambda d, key: d.get(key, {}) if isinstance(d, dict) else {}, keys, data)

def main():
    parser = argparse.ArgumentParser(description='Visualize metrics from JSON data.')
    parser.add_argument('--metric', nargs='+', action='append', default=[['validationLoss'], ['validationPerplexity']],
                        help='Metrics to visualize. Can be specified multiple times.')
    parser.add_argument('--label', nargs='+', action='store', default=[],
                        help='Additional metrics to include in the label. Can be specified multiple times.')
    args = parser.parse_args()

    data = load_data('metrics.json')
    
    for metric in args.metric:
        metric_key = metric[-1]
        metric_name = ' '.join(word for word in metric_key.split('_'))
        plot_metric(data, metric_key, metric_name, args.label)
        print(f"'{metric_name}' visualization saved as 'metrics_{metric_key}.png'")

if __name__ == "__main__":
    main()