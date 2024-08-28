import json
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def process_run_data(run, metric_key):
    validate_every = run['validateEvery']
    current_step = run['step']
    metric_values = run[metric_key][::-1]  # Reverse the array
    
    # Calculate the number of steps based on the length of metric_values
    num_steps = len(metric_values)
    
    # Calculate the last validation step (align to validation interval)
    last_validation_step = current_step - (current_step % validate_every)
    
    # Generate steps starting from the last validation step and going backwards
    steps = [last_validation_step - (validate_every * i) for i in range(num_steps)][::-1]
    
    return steps, metric_values

def plot_metric(data, metric_key, metric_name):
    plt.figure(figsize=(14, 8))  # Increased figure size
    sns.set_style("whitegrid")

    one_million = 1_000_000

    for run in data:
        steps, metric_values = process_run_data(run, metric_key)
        total_params = run['totalParams']
        label = f"v{run['version']} | {total_params / one_million:.2f}M | {run['runId']}\n{run['class']}"
        plt.plot(steps, metric_values, marker='o', label=label)

    plt.title(f"{metric_name} Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    plt.yscale('log')  # Use log scale for y-axis to better visualize small differences

    # Adjust legend
    # plt.legend(title="Run Info", title_fontsize=12, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(title="Run Info", title_fontsize=12)
    
    plt.yscale('log')  # Use log scale for y-axis to better visualize small differences
    plt.tight_layout()
    plt.savefig(f'metrics_{metric_key.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    data = load_data('metrics.json')
    
    # Plot Validation Loss
    plot_metric(data, 'validationLoss', 'Validation Loss')
    print("Validation Loss visualization saved as 'metrics_validationloss.png'")
    
    # Plot Perplexity
    plot_metric(data, 'validationPerplexity', 'Validation Perplexity')
    print("Validation Perplexity visualization saved as 'metrics_validationperplexity.png'")

if __name__ == "__main__":
    main()