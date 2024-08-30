import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from functools import reduce
import re
import numpy as np
from matplotlib.ticker import ScalarFormatter, LogLocator, AutoLocator

def plot_metric(data, metric_key, metric_name, label_metrics):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    one_million_bytes = 1_000_000
    all_values = []

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
        plt.plot(steps, metric_values, marker='o', label=label)
        all_values.extend([v for v in metric_values if not np.isnan(v)])

    formatted_name = split_variable(metric_name)
    plt.title(f"{formatted_name} Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(formatted_name, fontsize=12)

    plt.legend(title="Run Info", title_fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), frameon=True, fancybox=True, shadow=True)

    # Remove extreme outliers for better scaling
    filtered_values = remove_outliers(all_values)

    if len(filtered_values) > 0:
        vmin, vmax = np.min(filtered_values), np.max(filtered_values)
        plt.ylim(vmin, vmax)

        # Use symlog scale
        linthresh = max(abs(vmin), abs(vmax)) * 1e-3  # Adjust this value as needed
        plt.yscale('symlog', linthresh=linthresh)

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
    steps = [current_step - (validate_every * i) for i in range(num_steps)][::-1]
    
    return steps, metric_values

def get_nested_value(data, keys):
    if isinstance(keys, str):
        keys = keys.split('.')
    return reduce(lambda d, key: d.get(key, {}) if isinstance(d, dict) else {}, keys, data)

def remove_outliers(data, m=2.0):
    """Remove outliers using the Interquartile Range method."""
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values
    if len(data) == 0:
        return np.array([])
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (m * iqr)
    upper_bound = q3 + (m * iqr)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def main():
    parser = argparse.ArgumentParser(description='Visualize metrics from JSON data.')
    parser.add_argument('--metric', nargs='+', action='append', default=[['validationLoss'], ['validationPerplexity']],
                        help='Metrics to visualize. Can be specified multiple times.')
    parser.add_argument('--label', nargs='+', action='store', default=[['date']],
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