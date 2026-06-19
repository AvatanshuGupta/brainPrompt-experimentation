"""
TEST: What is the model actually learning?

Run this DURING training to understand:
1. Does model use graph structure or just memorize features?
2. Can model generalize to shuffled labels?
3. Is there a bug in loss/accuracy calculation?
"""

import torch
import numpy as np
from data.data import LoadData_llm
from torch.utils.data import DataLoader
from nets.load_net import gnn_model
import json

def test_model_with_scrambled_labels(model_name, dataset_name, config_file, device):
    """
    Train model with SCRAMBLED labels on small subset
    If accuracy is still high -> labels aren't actually being used
    """
    print("\n" + "="*70)
    print("TEST 1: Model with SCRAMBLED Labels")
    print("="*70)
    
    # Load config
    with open(config_file) as f:
        config = json.load(f)
    
    # Load small dataset
    dataset = LoadData_llm(dataset_name, threshold=0.3, node_feat_transform='pearson')
    trainset = dataset.train[0][:100]  # Just first 100 samples
    
    net_params = config['net_params']
    net_params['device'] = device
    net_params['node_num'] = dataset.node_num
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'].shape[1]
    net_params['edge_dim'] = dataset.all.graph_lists[0].edata['feat'][0].shape[0] \
        if 'feat' in dataset.all.graph_lists[0].edata else None
    net_params['n_classes'] = 2
    
    if model_name in ['BrainPromptG', 'BrainPromptC']:
        prompt_name = dataset_name.split('_')[0] + '_label.pt'
        label_embs = torch.stack(torch.load('data/prompts/label_prompts/' + prompt_name)).squeeze()
        net_params['label_embs'] = label_embs
    
    model = gnn_model(model_name, net_params, trainset)
    model = model.to(device)
    
    # Create normal and scrambled dataloaders
    train_loader_normal = DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=dataset.collate)
    
    # For scrambled: modify labels randomly
    trainset_scrambled = []
    for item in trainset:
        g, label, llm = item
        # Replace label with random class
        random_label = np.random.randint(0, 2)
        trainset_scrambled.append((g, random_label, llm))
    
    train_loader_scrambled = DataLoader(trainset_scrambled, batch_size=32, shuffle=False, collate_fn=dataset.collate)
    
    print(f"Training on {len(trainset)} samples")
    print("First 10 original labels:", [item[1] for item in trainset[:10]])
    print("First 10 scrambled labels:", [item[1] for item in trainset_scrambled[:10]])
    
    # Test accuracy on both
    model.eval()
    with torch.no_grad():
        acc_normal = 0
        acc_scrambled = 0
        
        for (batch_graphs_n, batch_labels_n, batch_llms_n), \
            (batch_graphs_s, batch_labels_s, batch_llms_s) in \
            zip(train_loader_normal, train_loader_scrambled):
            
            batch_graphs_n = batch_graphs_n.to(device)
            batch_x_n = batch_graphs_n.ndata['feat'].to(device)
            batch_e_n = batch_graphs_n.edata['feat'].to(device)
            batch_llms_n = batch_llms_n.to(device)
            batch_labels_n = batch_labels_n.to(device).long()
            
            batch_graphs_s = batch_graphs_s.to(device)
            batch_x_s = batch_graphs_s.ndata['feat'].to(device)
            batch_e_s = batch_graphs_s.edata['feat'].to(device)
            batch_llms_s = batch_llms_s.to(device)
            batch_labels_s = batch_labels_s.to(device).long()
            
            # Get predictions
            scores_n = model.forward(batch_graphs_n, batch_x_n, batch_e_n, batch_llms_n)
            scores_s = model.forward(batch_graphs_s, batch_x_s, batch_e_s, batch_llms_s)
            
            # Accuracy with true labels
            pred_n = torch.argmax(scores_n, dim=1)
            acc_normal += (pred_n == batch_labels_n).float().mean().item()
            
            # Accuracy with scrambled labels (should be ~50%)
            pred_s = torch.argmax(scores_s, dim=1)
            acc_scrambled += (pred_s == batch_labels_s).float().mean().item()
        
        acc_normal /= len(train_loader_normal)
        acc_scrambled /= len(train_loader_scrambled)
        
        print(f"\nResults (untrained model):")
        print(f"  Accuracy on ORIGINAL labels: {acc_normal:.4f}")
        print(f"  Accuracy on SCRAMBLED labels: {acc_scrambled:.4f}")
        
        if acc_scrambled > 0.55:
            print("\n🚨 CRITICAL ISSUE: Model performs well even on RANDOM labels!")
            print("   This suggests the model is NOT actually using labels")
            print("   Check: Accuracy calculation, loss function, or label assignment")
        else:
            print("\n✓ Normal - model performs poorly on random labels")


def test_feature_gradient_flow(model_name, dataset_name, config_file, device):
    """
    Check if gradients flow through graph features
    If not, model isn't learning from graph structure
    """
    print("\n" + "="*70)
    print("TEST 2: Gradient Flow Through Graph Features")
    print("="*70)
    
    with open(config_file) as f:
        config = json.load(f)
    
    dataset = LoadData_llm(dataset_name, threshold=0.3, node_feat_transform='pearson')
    trainset = dataset.train[0][:20]  # Small sample
    
    net_params = config['net_params']
    net_params['device'] = device
    net_params['node_num'] = dataset.node_num
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'].shape[1]
    net_params['edge_dim'] = dataset.all.graph_lists[0].edata['feat'][0].shape[0] \
        if 'feat' in dataset.all.graph_lists[0].edata else None
    net_params['n_classes'] = 2
    
    if model_name in ['BrainPromptG', 'BrainPromptC']:
        prompt_name = dataset_name.split('_')[0] + '_label.pt'
        label_embs = torch.stack(torch.load('data/prompts/label_prompts/' + prompt_name)).squeeze()
        net_params['label_embs'] = label_embs
    
    model = gnn_model(model_name, net_params, trainset)
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader = DataLoader(trainset, batch_size=32, shuffle=False, collate_fn=dataset.collate)
    
    print("Training for 1 batch, tracking gradient flow...")
    
    for batch_graphs, batch_labels, batch_llms in train_loader:
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_llms = batch_llms.to(device)
        batch_labels = batch_labels.to(device).long()
        
        # Check gradients at input
        batch_x.requires_grad_(True)
        batch_llms_copy = batch_llms.clone().detach().requires_grad_(True)
        
        scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms_copy)
        loss = model.loss(scores, batch_labels)
        loss.backward()
        
        # Check gradient flow
        print(f"\nBatch loss: {loss.item():.4f}")
        
        # Find gradient magnitudes for different components
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-6:
                    print(f"  {name[:50]:50s} | grad_norm: {grad_norm:.6f}")
        
        break  # Only one batch
    
    print("\nIf most parameters have zero gradients, the model isn't learning!")


def test_model_complexity_vs_performance(model_name, dataset_name, config_file, device):
    """
    Check if model is just too complex for simple problem
    """
    print("\n" + "="*70)
    print("TEST 3: Model Capacity Analysis")
    print("="*70)
    
    with open(config_file) as f:
        config = json.load(f)
    
    dataset = LoadData_llm(dataset_name, threshold=0.3, node_feat_transform='pearson')
    trainset = dataset.train[0]
    
    net_params = config['net_params']
    net_params['device'] = device
    net_params['node_num'] = dataset.node_num
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'].shape[1]
    net_params['edge_dim'] = dataset.all.graph_lists[0].edata['feat'][0].shape[0] \
        if 'feat' in dataset.all.graph_lists[0].edata else None
    net_params['n_classes'] = 2
    
    if model_name in ['BrainPromptG', 'BrainPromptC']:
        prompt_name = dataset_name.split('_')[0] + '_label.pt'
        label_embs = torch.stack(torch.load('data/prompts/label_prompts/' + prompt_name)).squeeze()
        net_params['label_embs'] = label_embs
    
    model = gnn_model(model_name, net_params, trainset)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_name}")
    print(f"Dataset size: {len(trainset)} samples")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Params per sample ratio: {trainable_params / len(trainset):.2f}")
    
    if trainable_params / len(trainset) > 1:
        print(f"\n⚠️  Model is VERY COMPLEX relative to dataset!")
        print(f"   {trainable_params:,} params for {len(trainset)} samples")
        print(f"   Ratio > 1 means high overfitting risk")
    else:
        print(f"\n✓ Model capacity seems reasonable")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='BrainPromptG')
    parser.add_argument('--dataset', default='abide_full_AAL116')
    parser.add_argument('--config', default='configs/abide_full_AAL116/TUs_graph_classification_BrainPromptG_abide_full_AAL116_100k.json')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("MODEL BEHAVIOR DIAGNOSTIC")
    print("="*70)
    
    test_model_complexity_vs_performance(args.model, args.dataset, args.config, device)
    test_model_with_scrambled_labels(args.model, args.dataset, args.config, device)
    # test_feature_gradient_flow(args.model, args.dataset, args.config, device)  # Optional
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("""
If Test 1 shows high accuracy even with SCRAMBLED labels:
  → Model is NOT learning from the actual labels
  → This is a MODEL/TRAINING BUG, not data leakage
  → Check: Loss function, label assignment, training loop
  
If params > dataset_size:
  → Model can memorize entire training set
  → Add regularization (dropout, weight decay, early stopping)
    """)
    print("="*70 + "\n")
