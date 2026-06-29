import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_perplexity(json_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    base_ppl = data['base']
    qlora_ppl = data['qlora']
    
    # Create DataFrame for Seaborn
    df = pd.DataFrame({
        'Perplexity': base_ppl + qlora_ppl,
        'Model': ['Base (No LoRA)'] * len(base_ppl) + ['QLoRA Fine-tuned'] * len(qlora_ppl)
    })
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Violin Plot / KDE
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Model', y='Perplexity', data=df, palette=['#ff7f0e', '#1f77b4'], inner="quartile")
    plt.title("Distribution of Perplexity Across State Prompts", pad=15, fontweight="bold")
    plt.ylabel("Perplexity (Lower is Better)")
    plt.yscale("log") # Log scale for better visibility if variance is high
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_violin.png"), dpi=300)
    plt.close()
    
    # 2. Bar Plot with 95% Confidence Intervals
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Model', y='Perplexity', data=df, palette=['#ff7f0e', '#1f77b4'], capsize=.1, errorbar=('ci', 95))
    plt.title("Average Perplexity with 95% Confidence Intervals", pad=15, fontweight="bold")
    plt.ylabel("Average Perplexity")
    
    # Annotate with mean values
    means = df.groupby('Model')['Perplexity'].mean()
    for i, model in enumerate(['Base (No LoRA)', 'QLoRA Fine-tuned']):
        plt.text(i, means[model] + 0.1, f"{means[model]:.2f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_bar_ci.png"), dpi=300)
    plt.close()
    
    print(f"Saved perplexity plots to {output_dir}")

if __name__ == "__main__":
    json_path = "plots/comparison/per_prompt_perplexity.json"
    if os.path.exists(json_path):
        plot_perplexity(json_path, "plots/comparison")
    else:
        print(f"Error: {json_path} not found. Run evaluate_perplexity.py first.")
