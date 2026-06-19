"""
SIMPLIFIED DIAGNOSTIC: Just check model capacity
"""

import torch
import numpy as np
import json

def count_model_parameters(model_name, config_file):
    """
    Quick way to check model size without fully initializing
    """
    print("\n" + "="*70)
    print("MODEL CAPACITY ANALYSIS")
    print("="*70)
    
    with open(config_file) as f:
        config = json.load(f)
    
    net_params = config['net_params']
    
    # Manually calculate expected parameters based on architecture
    L = net_params.get('L', 4)
    hidden_dim = net_params.get('hidden_dim', 64)
    in_dim = 116  # For AAL116
    n_classes = 2
    node_num = 116
    
    # Estimate for BrainPromptG
    model_name = 'BrainPromptG'
    
    print(f"Model: {model_name}")
    print(f"Architecture parameters:")
    print(f"  Layers (L): {L}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Input dim: {in_dim}")
    print(f"  Output classes: {n_classes}")
    print(f"  Node count: {node_num}")
    
    # Rough estimation
    # Each layer: input*hidden + hidden*hidden + biases
    # For graph neural network with ~4 layers
    estimated_params = 0
    
    # Input projection
    estimated_params += in_dim * hidden_dim  # 116 * 64 = 7,424
    
    # Hidden layers: L layers
    for i in range(L):
        estimated_params += hidden_dim * hidden_dim * 2  # 64*64*2 per layer for message passing
    
    # Output projection
    estimated_params += hidden_dim * n_classes  # 64 * 2 = 128
    
    # Readout/pooling layers (rough)
    estimated_params += hidden_dim * 2 * hidden_dim  # Readout layer
    
    print(f"\nEstimated parameters (rough): {estimated_params:,}")
    print(f"Dataset size (train): 818 samples")
    print(f"Params per sample: {estimated_params / 818:.2f}")
    
    if estimated_params / 818 > 1.0:
        print(f"\n⚠️  OVERFITTING RISK: Model can potentially memorize entire dataset!")
        print(f"   Ratio = {estimated_params / 818:.2f} (> 1.0 is concerning)")
    else:
        print(f"\n✓ Model capacity seems reasonable (ratio < 1.0)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/abide_full_AAL116/TUs_graph_classification_BrainPromptG_abide_full_AAL116_100k.json')
    args = parser.parse_args()
    
    count_model_parameters('BrainPromptG', args.config)
    
    print("\n" + "="*70)
    print("SOLUTION")
    print("="*70)
    print("""
If params/sample ratio > 1.0:
  1. Reduce model capacity (reduce hidden_dim or L)
  2. Increase regularization:
     - Increase dropout from 0.3 to 0.7
     - Increase weight_decay from 0.0 to 0.01-0.1
  3. Add early stopping (reduce max_time or increase patience)
  4. Use cross-validation properly (you're doing 10-fold, good!)

The model may be achieving 100% by MEMORIZING training data,
not by learning generalizable patterns.
    """)
    print("="*70 + "\n")
