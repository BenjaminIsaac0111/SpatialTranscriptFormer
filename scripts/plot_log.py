import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_log(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Identify categories
    loss_cols = [c for c in df.columns if 'loss' in c.lower()]
    metric_cols = [c for c in df.columns if 'pcc' in c.lower() or 'pcc_mean' in c.lower()]
    lr_cols = [c for c in df.columns if 'lr' in c.lower()]
    
    fig, axes = plt.subplots(2 if len(metric_cols) > 0 else 1, 1, figsize=(10, 10), squeeze=False)
    
    # Plot Losses
    ax_loss = axes[0, 0]
    for col in loss_cols:
        ax_loss.plot(df['epoch'], df[col], label=col)
    ax_loss.set_title('Training Losses')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.set_yscale('log')
    
    # Plot Metrics
    if len(metric_cols) > 0:
        ax_metric = axes[1, 0]
        for col in metric_cols:
            ax_metric.plot(df['epoch'], df[col], label=col)
        ax_metric.set_title('Training Metrics')
        ax_metric.set_xlabel('Epoch')
        ax_metric.set_ylabel('Score')
        ax_metric.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    csv_file = r"z:\Projects\SpatialTranscriptFormer\runs\stf_pathway_hybrid\training_log.csv"
    out_file = r"z:\Projects\SpatialTranscriptFormer\runs\stf_pathway_hybrid\training_history.png"
    plot_training_log(csv_file, out_file)
