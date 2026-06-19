"""
DEEP LEAKAGE DIAGNOSTIC
This script identifies whether 100% accuracy comes from:
1. Graph structure leakage
2. Node features encoding labels
3. Train/test overlap
4. Model architecture bias
"""

import torch
import numpy as np
from data.data import LoadData_llm
import hashlib

def check_train_test_overlap(dataset_name, threshold=0.3, node_feat_transform='pearson'):
    """
    Check if same samples appear in train and test sets
    """
    print("\n" + "="*70)
    print("TRAIN/TEST DATA OVERLAP CHECK")
    print("="*70)
    
    dataset = LoadData_llm(dataset_name, threshold=threshold, node_feat_transform=node_feat_transform)
    
    trainset = dataset.train[0]
    testset = dataset.test[0]
    
    # Create hashes of graph features for exact comparison
    train_hashes = set()
    train_graphs = []
    
    for item in trainset:
        if len(item) == 3:
            g, label, llm = item
        else:
            g = item[0]
        
        feat_bytes = g.ndata['feat'].cpu().numpy().tobytes()
        feat_hash = hashlib.md5(feat_bytes).hexdigest()
        train_hashes.add(feat_hash)
        train_graphs.append((feat_hash, g.ndata['feat'].cpu().numpy()))
    
    test_overlaps = 0
    test_match_features = []
    
    for item in testset:
        if len(item) == 3:
            g, label, llm = item
        else:
            g = item[0]
        
        feat_bytes = g.ndata['feat'].cpu().numpy().tobytes()
        feat_hash = hashlib.md5(feat_bytes).hexdigest()
        
        if feat_hash in train_hashes:
            test_overlaps += 1
            test_match_features.append(g.ndata['feat'].cpu().numpy())
    
    print(f"Train set: {len(trainset)} samples")
    print(f"Test set: {len(testset)} samples")
    print(f"Exact feature matches (train ∩ test): {test_overlaps}")
    
    if test_overlaps > 0:
        print(f"🚨 CRITICAL LEAKAGE: {test_overlaps/len(testset)*100:.1f}% of test set also in training!")
    else:
        print("✓ No exact feature overlap between train and test")


def check_node_feature_separability(dataset_name, threshold=0.3, node_feat_transform='pearson'):
    """
    Check if node features alone can perfectly separate classes
    """
    print("\n" + "="*70)
    print("NODE FEATURE SEPARABILITY CHECK")
    print("="*70)
    
    dataset = LoadData_llm(dataset_name, threshold=threshold, node_feat_transform=node_feat_transform)
    
    trainset = dataset.train[0]
    testset = dataset.test[0]
    
    for split_name, split_data in [("Train", trainset), ("Test", testset)]:
        print(f"\n{split_name} Set:")
        
        all_features = []
        all_labels = []
        
        for item in split_data:
            if len(item) == 3:
                g, label, llm = item
            else:
                g, label = item[0], item[1]
            
            # Get graph-level feature: mean of all node features
            node_feats = g.ndata['feat'].cpu().numpy()
            graph_feat = node_feats.mean(axis=0)  # Average pooling
            all_features.append(graph_feat)
            all_labels.append(label)
        
        features = np.array(all_features)
        labels = np.array(all_labels)
        
        # Simple linear separability test using distance to class means
        cls0_feats = features[labels == 0]
        cls1_feats = features[labels == 1]
        
        center_0 = cls0_feats.mean(axis=0)
        center_1 = cls1_feats.mean(axis=0)
        
        # Predict by distance to class centers
        from sklearn.metrics.pairwise import euclidean_distances
        dists_to_0 = euclidean_distances(features, center_0.reshape(1, -1)).ravel()
        dists_to_1 = euclidean_distances(features, center_1.reshape(1, -1)).ravel()
        
        predictions = (dists_to_0 > dists_to_1).astype(int)
        accuracy = (predictions == labels).mean()
        
        print(f"  Accuracy using mean node features: {accuracy:.4f}")
        
        if accuracy > 0.9:
            print(f"  ⚠️  HIGHLY SEPARABLE: Node features encode strong class signal!")
        
        # Check mean feature difference per class
        mean_diff = np.linalg.norm(center_0 - center_1)
        print(f"  Distance between class centers: {mean_diff:.4f}")


def check_graph_topology_leakage(dataset_name, threshold=0.3, node_feat_transform='pearson'):
    """
    Check if graph topology (edges) differ between classes
    """
    print("\n" + "="*70)
    print("GRAPH TOPOLOGY LEAKAGE CHECK")
    print("="*70)
    
    dataset = LoadData_llm(dataset_name, threshold=threshold, node_feat_transform=node_feat_transform)
    
    trainset = dataset.train[0]
    
    class_0_edges = []
    class_1_edges = []
    
    for item in trainset:
        if len(item) == 3:
            g, label, llm = item
        else:
            g, label = item[0], item[1]
        
        edge_count = g.number_of_edges()
        
        if label == 0:
            class_0_edges.append(edge_count)
        else:
            class_1_edges.append(edge_count)
    
    print(f"Class 0 - Edge count stats:")
    print(f"  Mean: {np.mean(class_0_edges):.1f}, Std: {np.std(class_0_edges):.1f}")
    print(f"Class 1 - Edge count stats:")
    print(f"  Mean: {np.mean(class_1_edges):.1f}, Std: {np.std(class_1_edges):.1f}")
    
    edge_diff = abs(np.mean(class_0_edges) - np.mean(class_1_edges))
    print(f"Difference in edge counts: {edge_diff:.1f}")
    
    if edge_diff > 50:
        print("⚠️  Different graph topologies between classes - potential signal!")
    else:
        print("✓ Graph topologies similar between classes")


def test_with_random_labels(model, device, data_loader):
    """
    If model achieves good accuracy even with random labels,
    something is seriously wrong (e.g., not using labels at all)
    """
    print("\n" + "="*70)
    print("SANITY CHECK: Model with Random Labels")
    print("="*70)
    
    model.eval()
    acc_with_random_labels = 0
    nb_data = 0
    
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_llms = batch_llms.to(device)
            
            # Replace with random labels
            batch_labels_random = torch.randint(0, 2, (batch_labels.shape[0],)).to(device).long()
            
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
            
            # Calculate accuracy with random labels
            pred_labels = torch.argmax(batch_scores, dim=1)
            acc_with_random_labels += (pred_labels == batch_labels_random).float().mean().item()
            nb_data += 1
    
    avg_acc = acc_with_random_labels / nb_data
    print(f"Model accuracy with RANDOM labels: {avg_acc:.4f}")
    
    if avg_acc > 0.6:
        print("🚨 CRITICAL: Model performs well even with random labels!")
        print("   This means the model is NOT actually learning the labels!")
        print("   Check: Is the loss function being called correctly?")
    else:
        print("✓ Model's random label performance is near 50% (expected)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='abide_full_AAL116', help='Dataset name')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--node_feat_transform', default='pearson')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DEEP DIAGNOSTIC: Finding the source of 100% accuracy")
    print("="*70)
    
    check_train_test_overlap(args.dataset, args.threshold, args.node_feat_transform)
    check_node_feature_separability(args.dataset, args.threshold, args.node_feat_transform)
    check_graph_topology_leakage(args.dataset, args.threshold, args.node_feat_transform)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Since LLM embeddings are NOT leaking (0% accuracy), the 100% accuracy is coming from:

1. GRAPH FEATURES: Node feature values encode the class signal
   → Check how node features are computed
   → Are they class-aware in generation?
   
2. GRAPH TOPOLOGY: Edge structure differs between classes
   → Check threshold application in graph construction
   → Is threshold different for healthy vs diseased?
   
3. TRAIN/TEST OVERLAP: Same samples appear in both splits
   → Check data splitting logic
   
4. MODEL ISSUE: Check training with random labels
   → If accuracy > 0.6, loss function is broken
   → Model is learning pseudo-labels, not actual labels

NEXT STEP: Run training with diagnostics to see which component
is driving 100% accuracy
    """)
    print("="*70 + "\n")
