"""
STANDALONE DIAGNOSTIC SCRIPT
Check if LLM embeddings encode label information (data leakage)
Run this BEFORE training to identify the root cause
"""

import torch
import numpy as np
from data.data import LoadData_llm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def check_llm_embedding_leakage(dataset_name, threshold=0.3, node_feat_transform='pearson'):
    """
    Comprehensive check for label leakage in LLM embeddings
    """
    print("\n" + "="*70)
    print(f"LLM EMBEDDING LEAKAGE DIAGNOSTIC FOR: {dataset_name}")
    print("="*70)
    
    # Load dataset
    dataset = LoadData_llm(dataset_name, threshold=threshold, node_feat_transform=node_feat_transform)
    
    # Use first fold
    trainset = dataset.train[0]
    valset = dataset.val[0]
    testset = dataset.test[0]
    
    for split_name, split_data in [("Train", trainset), ("Val", valset), ("Test", testset)]:
        print(f"\n{'='*70}")
        print(f"{split_name} Set Analysis")
        print(f"{'='*70}")
        
        all_llms = []
        all_labels = []
        
        for g, label, llm_emb in split_data:
            all_llms.append(llm_emb.numpy() if hasattr(llm_emb, 'numpy') else llm_emb)
            all_labels.append(label)
        
        llms = np.array(all_llms)
        labels = np.array(all_labels)
        
        print(f"Samples: {len(labels)}")
        print(f"LLM embedding shape: {llms.shape}")
        print(f"Label distribution: {np.bincount(labels.astype(int))}")
        
        # Analysis per class
        unique_labels = np.unique(labels)
        if len(unique_labels) == 2:
            cls0_llms = llms[labels == unique_labels[0]]
            cls1_llms = llms[labels == unique_labels[1]]
            
            print(f"\n--- Class {unique_labels[0]} ---")
            print(f"  Samples: {len(cls0_llms)}")
            print(f"  Mean embedding (first 5 dims): {cls0_llms.mean(axis=0)[:5]}")
            print(f"  Std embedding (first 5 dims): {cls0_llms.std(axis=0)[:5]}")
            print(f"  Min: {cls0_llms.min():.6f}, Max: {cls0_llms.max():.6f}")
            
            print(f"\n--- Class {unique_labels[1]} ---")
            print(f"  Samples: {len(cls1_llms)}")
            print(f"  Mean embedding (first 5 dims): {cls1_llms.mean(axis=0)[:5]}")
            print(f"  Std embedding (first 5 dims): {cls1_llms.std(axis=0)[:5]}")
            print(f"  Min: {cls1_llms.min():.6f}, Max: {cls1_llms.max():.6f}")
            
            # KEY TEST: Can we separate classes using ONLY embeddings?
            centers = np.array([cls0_llms.mean(axis=0), cls1_llms.mean(axis=0)])
            dists_to_0 = euclidean_distances(llms, centers[0:1])
            dists_to_1 = euclidean_distances(llms, centers[1:2])
            
            predictions_by_emb = (dists_to_0 < dists_to_1).ravel().astype(int)
            accuracy_by_emb = (predictions_by_emb == labels).mean()
            
            print(f"\n--- LEAKAGE TEST (Critical) ---")
            print(f"Accuracy using ONLY LLM embeddings + nearest center: {accuracy_by_emb:.4f}")
            
            if accuracy_by_emb > 0.95:
                print("🚨 CRITICAL LEAKAGE DETECTED!")
                print("    LLM embeddings encode class information perfectly.")
                print("    → Problem is in data generation, not the model")
                print("    → Check how 'label_prompts' are created")
                print("    → Are label descriptions specific to each class?")
            elif accuracy_by_emb > 0.80:
                print("⚠️  SIGNIFICANT LEAKAGE")
                print("    LLM embeddings have discriminative information")
            else:
                print("✓ ACCEPTABLE: LLM embeddings don't strongly encode labels")
            
            # Cosine similarity check
            mean_diff = np.linalg.norm(cls0_llms.mean(axis=0) - cls1_llms.mean(axis=0))
            print(f"\nMean embedding distance between classes: {mean_diff:.6f}")
            if mean_diff < 0.01:
                print("  ⚠️  Class means are nearly identical (suspicious)")
            
            # Check if embeddings are all the same within class
            within_class_std = np.concatenate([cls0_llms.std(axis=0), cls1_llms.std(axis=0)]).mean()
            print(f"Mean std within each class: {within_class_std:.6f}")
            if within_class_std < 0.001:
                print("  ⚠️  Embeddings have very low variance (all samples identical?)")
        
        print()


def check_graph_structure(dataset_name, threshold=0.3, node_feat_transform='pearson'):
    """
    Check if graphs vary enough or if they're all similar
    """
    print("\n" + "="*70)
    print("GRAPH STRUCTURE ANALYSIS")
    print("="*70)
    
    dataset = LoadData_llm(dataset_name, threshold=threshold, node_feat_transform=node_feat_transform)
    trainset = dataset.train[0]
    
    feat_norms = []
    edge_counts = []
    node_counts = []
    
    # Collect first 20 samples
    for i, item in enumerate(trainset[:20]):  # Check first 20
        if len(item) == 3:
            g, label, llm = item
        else:
            g = item[0]  # Just get the graph
        
        feat_norm = g.ndata['feat'].norm().item()
        edge_count = g.number_of_edges()
        node_count = g.number_of_nodes()
        feat_norms.append(feat_norm)
        edge_counts.append(edge_count)
        node_counts.append(node_count)
    
    print(f"Feature norm stats (first 20 graphs):")
    print(f"  Mean: {np.mean(feat_norms):.4f}, Std: {np.std(feat_norms):.4f}")
    print(f"Edge count stats (first 20 graphs):")
    print(f"  Mean: {np.mean(edge_counts):.1f}, Std: {np.std(edge_counts):.1f}")
    print(f"Node count stats (first 20 graphs):")
    print(f"  Mean: {np.mean(node_counts):.1f}, Std: {np.std(node_counts):.1f}")
    
    if np.std(feat_norms) < 0.01:
        print("  ⚠️  Graph features are nearly identical across samples!")
    if np.std(edge_counts) < 1:
        print("  ⚠️  Graph structures are nearly identical across samples!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='abide_full_AAL116', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--node_feat_transform', default='pearson')
    args = parser.parse_args()
    
    check_llm_embedding_leakage(args.dataset, args.threshold, args.node_feat_transform)
    check_graph_structure(args.dataset, args.threshold, args.node_feat_transform)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. If LLM accuracy > 0.95: Problem is in label_prompts generation")
    print("   → Check data/prompts/label_prompts/ files")
    print("   → Are prompts different for each class?")
    print("\n2. If graph features all identical: Problem is in graph generation")
    print("   → Check data.py graph construction")
    print("\n3. Run training with new debugging code to see diagnostic output")
    print("="*70 + "\n")
