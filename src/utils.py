"""Utility functions"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(hyperfocus_losses, baseline_losses, save_path='training_comparison.png'):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Raw losses
    ax1.plot(hyperfocus_losses, label='HyperFocus-5M', linewidth=2, alpha=0.8)
    ax1.plot(baseline_losses, label='Baseline', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Smoothed
    window = 20
    if len(hyperfocus_losses) > window:
        hf_smooth = np.convolve(hyperfocus_losses, np.ones(window)/window, mode='valid')
        bl_smooth = np.convolve(baseline_losses, np.ones(window)/window, mode='valid')
        
        ax2.plot(hf_smooth, label='HyperFocus (smoothed)', linewidth=2)
        ax2.plot(bl_smooth, label='Baseline (smoothed)', linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Loss (smoothed)')
        ax2.set_title('Smoothed Loss Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}")
    
    return fig


def visualize_routing(model, tokenizer, text, device='cpu'):
    """Visualize token routing behavior"""
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    with torch.no_grad():
        _, routing_scores = model(input_ids)
    
    avg_scores = torch.stack(routing_scores).mean(dim=0)[0].cpu().numpy()
    
    plt.figure(figsize=(14, 5))
    threshold = np.percentile(avg_scores, 75)
    colors = ['red' if s > threshold else 'blue' for s in avg_scores]
    
    plt.bar(range(len(tokens)), avg_scores, color=colors, alpha=0.6)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.ylabel('Routing Score')
    plt.title('Token Routing (Red=Deep Path, Blue=Shallow)')
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return plt.gcf()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
