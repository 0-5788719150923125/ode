import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from functools import reduce
import re

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def split_variable(string):
    # Split the string on capital letters
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', string)
    
    # Join the words with spaces and capitalize each word
    return ' '.join(word.capitalize() for word in words)

def process_run_data(run, metric_key):
    validate_every = run['validateEvery']
    current_step = run['step']
    metric_values = run[metric_key][::-1]  # Reverse the array
    
    num_steps = len(metric_values)
    last_validation_step = current_step - (current_step % validate_every)
    steps = [last_validation_step - (validate_every * i) for i in range(num_steps)][::-1]
    
    return steps, metric_values

def get_nested_value(data, keys):
    if isinstance(keys, str):
        return data.get(keys)
    return reduce(lambda d, key: d.get(key, {}) if isinstance(d, dict) else {}, keys, data)

def plot_metric(data, metric_key, metric_name, label_metrics):
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    one_million_bytes = 1_000_000

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

    formatted_name = split_variable(metric_name)
    plt.title(f"{formatted_name} Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(formatted_name, fontsize=12)

    plt.legend(title="Run Info", title_fontsize=12)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'metrics_{metric_key}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize metrics from JSON data.')
    parser.add_argument('--metric', nargs='+', action='append', default=[['validationLoss'], ['validationPerplexity']],
                        help='Metrics to visualize. Can be specified multiple times.')
    parser.add_argument('--label', nargs='+', action='store', default=[['lossFunction', 'name']],
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