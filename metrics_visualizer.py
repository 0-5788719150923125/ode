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
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    for run in data:
        steps, metric_values = process_run_data(run, metric_key)
        plt.plot(steps, metric_values, marker='o', label=f"v{run['version']} ({run['runId']})")

    plt.title(f"{metric_name} Over Time", fontsize=16)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend(title="Run ID", title_fontsize=12)
    plt.yscale('log')  # Use log scale for y-axis to better visualize small differences
    plt.tight_layout()
    plt.savefig(f'metrics_{metric_key.lower()}.png', dpi=300)
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

# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime

# # Load the JSON data
# with open('metrics.json', 'r') as f:
#     data = json.load(f)

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Convert timestamp to datetime
# df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# # Sort by datetime
# df = df.sort_values('datetime')

# # Helper function to safely get the first element of a list or return None
# def safe_first(x):
#     return x[0] if isinstance(x, list) and len(x) > 0 else None

# # Create visualizations
# plt.figure(figsize=(15, 10))

# # 1. Training Loss over time
# plt.subplot(2, 2, 1)
# for run_id in df['runId'].unique():
#     run_data = df[df['runId'] == run_id]
#     loss_values = run_data['loss'].apply(safe_first)
#     plt.plot(run_data['datetime'], loss_values, label=f'Run {run_id}')
# plt.title('Training Loss over Time')
# plt.xlabel('Time')
# plt.ylabel('Loss')
# plt.legend()

# # 2. Validation Loss over time
# plt.subplot(2, 2, 2)
# for run_id in df['runId'].unique():
#     run_data = df[df['runId'] == run_id]
#     val_loss_values = run_data['validationLoss'].apply(safe_first)
#     plt.plot(run_data['datetime'], val_loss_values, label=f'Run {run_id}')
# plt.title('Validation Loss over Time')
# plt.xlabel('Time')
# plt.ylabel('Validation Loss')
# plt.legend()

# # 3. Learning Rate vs Loss
# plt.subplot(2, 2, 3)
# for run_id in df['runId'].unique():
#     run_data = df[df['runId'] == run_id]
#     learning_rates = run_data['optimizer'].apply(lambda x: x['learningRate'] if isinstance(x, dict) else None)
#     loss_values = run_data['loss'].apply(safe_first)
#     plt.scatter(learning_rates, loss_values, label=f'Run {run_id}')
# plt.title('Learning Rate vs Loss')
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.xscale('log')
# plt.legend()

# # 4. Validation Perplexity over time
# plt.subplot(2, 2, 4)
# for run_id in df['runId'].unique():
#     run_data = df[df['runId'] == run_id]
#     perplexity_values = run_data['validationPerplexity'].apply(safe_first)
#     plt.plot(run_data['datetime'], perplexity_values, label=f'Run {run_id}')
# plt.title('Validation Perplexity over Time')
# plt.xlabel('Time')
# plt.ylabel('Validation Perplexity')
# plt.legend()

# plt.tight_layout()
# plt.savefig('training_metrics_visualization.png')
# plt.close()

# # 5. Heatmap of architecture parameters vs final validation loss
# latest_runs = df.groupby('runId').last().reset_index()
# arch_params = pd.json_normalize(latest_runs['architecture'])
# latest_runs['final_val_loss'] = latest_runs['validationLoss'].apply(safe_first)
# heatmap_data = pd.concat([arch_params, latest_runs['final_val_loss']], axis=1)
# correlation_matrix = heatmap_data.corr()

# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
# plt.title('Correlation of Architecture Parameters with Validation Loss')
# plt.savefig('architecture_correlation_heatmap.png')
# plt.close()

# print("Visualizations saved as 'training_metrics_visualization.png' and 'architecture_correlation_heatmap.png'")