"""Test if graph features alone can achieve 100% accuracy"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data import LoadData_llm
import dgl
import numpy as np

# Load dataset
dataset = LoadData_llm(DATASET_NAME='abide_full_AAL116')

# Use 10-fold CV fold 0
fold = 0
train_idx = np.loadtxt(f'data/abide_full_AAL116/train.index').astype(int)
val_idx = np.loadtxt(f'data/abide_full_AAL116/val.index').astype(int)
test_idx = np.loadtxt(f'data/abide_full_AAL116/test.index').astype(int)

trainset = [dataset[i] for i in train_idx]
valset = [dataset[i] for i in val_idx]
testset = [dataset[i] for i in test_idx]

print(f"Train: {len(trainset)}, Val: {len(valset)}, Test: {len(testset)}")

# Simple MLP classifier using only graph node features
class GraphOnlyClassifier(nn.Module):
    def __init__(self, input_dim=116, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
    
    def forward(self, h):
        # h: [batch_size, num_nodes, input_dim]
        # Average pooling over nodes
        x = h.mean(dim=1)  # [batch_size, input_dim]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Prepare data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphOnlyClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("\n========== TRAINING ==========")

for epoch in range(10):
    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for g, y, _ in trainset:
        h = g.ndata['feat'].to(device)
        y_tensor = torch.tensor([y], dtype=torch.long).to(device)
        
        h = h.unsqueeze(0)  # Add batch dimension
        logits = model(h)
        loss = criterion(logits, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = logits.argmax(dim=1)
        train_correct += (pred == y_tensor).sum().item()
        train_total += 1
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for g, y, _ in valset:
            h = g.ndata['feat'].to(device).unsqueeze(0)
            logits = model(h)
            pred = logits.argmax(dim=1)
            val_correct += (pred == y).sum().item()
            val_total += 1
    
    # Test
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for g, y, _ in testset:
            h = g.ndata['feat'].to(device).unsqueeze(0)
            logits = model(h)
            pred = logits.argmax(dim=1)
            test_correct += (pred == y).sum().item()
            test_total += 1
    
    train_acc = train_correct / train_total
    val_acc = val_correct / val_total
    test_acc = test_correct / test_total
    
    print(f"Epoch {epoch}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, test_acc={test_acc:.3f}, loss={train_loss/train_total:.4f}")

print("\n✅ If test accuracy is high, the label leakage is in graph features!")
print("✅ If test accuracy is low, then LLM embeddings are the issue!")
