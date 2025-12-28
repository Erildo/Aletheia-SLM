#!/usr/bin/env python3
"""
Main training script for HyperFocus prototype - FIXED VERSION
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import argparse

from src.model import HyperFocusLM, BaselineTransformer
from src.train import train_model, generate_text
from src.utils import plot_training_curves, count_parameters


def main(args):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer FIRST to get correct vocab size
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # FIX: Use actual tokenizer vocab size
    actual_vocab_size = len(tokenizer)
    print(f"Tokenizer vocabulary size: {actual_vocab_size}")
    
    # Override the vocab_size argument
    args.vocab_size = actual_vocab_size
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{args.num_samples}]")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )
    
    print("Tokenizing...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    train_loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Dataset ready: {len(tokenized)} samples, {len(train_loader)} batches")
    
    # Initialize models with CORRECT vocab size
    print(f"\nInitializing models with vocab_size={args.vocab_size}...")
    model = HyperFocusLM(
        vocab_size=args.vocab_size,  # Now matches tokenizer
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    
    baseline = BaselineTransformer(
        vocab_size=args.vocab_size,  # Now matches tokenizer
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads
    ).to(device)
    
    print(f"HyperFocus params: {count_parameters(model) / 1e6:.2f}M")
    print(f"Baseline params: {count_parameters(baseline) / 1e6:.2f}M")
    
    # Train models
    print("\n" + "="*70)
    print("TRAINING HYPERFOCUS")
    print("="*70)
    hf_losses = train_model(model, train_loader, args.epochs, args.lr, device, "HyperFocus")
    
    print("\n" + "="*70)
    print("TRAINING BASELINE")
    print("="*70)
    bl_losses = train_model(baseline, train_loader, args.epochs, args.lr, device, "Baseline")
    
    # Save results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"HyperFocus final loss: {hf_losses[-1]:.4f}")
    print(f"Baseline final loss: {bl_losses[-1]:.4f}")
    improvement = ((bl_losses[-1] - hf_losses[-1]) / bl_losses[-1]) * 100
    print(f"Improvement: {improvement:+.2f}%")
    
    # Plot
    plot_training_curves(hf_losses, bl_losses, 'results/training_comparison.png')
    
    # Save models
    torch.save(model.state_dict(), 'models/hyperfocus_5m.pt')
    torch.save(baseline.state_dict(), 'models/baseline_5m.pt')
    print("\n✓ Models saved to models/")
    
    # Generate samples
    print("\n" + "="*70)
    print("SAMPLE GENERATIONS")
    print("="*70)
    prompts = ["Once upon a time", "The little girl", "In a magical forest"]
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            print(f"HyperFocus: {generate_text(model, tokenizer, prompt, device=device)}")
        except Exception as e:
            print(f"HyperFocus generation failed: {e}")
        try:
            print(f"Baseline: {generate_text(baseline, tokenizer, prompt, device=device)}")
        except Exception as e:
            print(f"Baseline generation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)  # Reduced for CPU
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=50257)  # GPT-2 default, will be overridden
    parser.add_argument("--max_length", type=int, default=128)
    
    args = parser.parse_args()
    
    # Create directories
    import os
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    main(args)
