"""Test if LLM embeddings alone can achieve perfect classification"""
import torch
import torch.nn as nn
from data.data import LoadData_llm
import json

# Load config
config_path = "configs/abide_full_AAL116/TUs_graph_classification_BrainPromptG_abide_full_AAL116_100k.json"
with open(config_path, 'r') as f:
    config = json.load(f)

# Load dataset
dataset = LoadData_llm(config['dataset'], threshold=config.get('threshold', 0.3), 
                        node_feat_transform=config.get('node_feat_transform', 'original'))

# Use split 0 for testing
train_set = dataset.train[0]
val_set = dataset.val[0]
test_set = dataset.test[0]

train_loader = torch.utils.data.DataLoader(train_set, batch_size=34, shuffle=True, collate_fn=dataset.collate)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=34, shuffle=False, collate_fn=dataset.collate)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=34, shuffle=False, collate_fn=dataset.collate)

# Simple classifier on LLM embeddings alone
class LLMClassifier(nn.Module):
    def __init__(self, llm_dim=2048, n_classes=2):
        super().__init__()
        self.linear1 = nn.Linear(llm_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, n_classes)
    
    def forward(self, llm):
        x = torch.relu(self.linear1(llm))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = LLMClassifier().to('cuda:0')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
criterion = nn.CrossEntropyLoss()

print("[LLM-ONLY TEST] Training classifier on LLM embeddings ALONE (no graph features)")
print("=" * 70)

# Train for 5 epochs
for epoch in range(5):
    # Train
    train_correct = 0
    train_total = 0
    train_loss = 0
    for batch_graphs, batch_labels, batch_llms in train_loader:
        batch_labels = batch_labels.long().to('cuda:0')
        batch_llms = batch_llms.to('cuda:0').squeeze(1)
        
        optimizer.zero_grad()
        outputs = model(batch_llms)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == batch_labels).sum().item()
        train_total += batch_labels.size(0)
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_graphs, batch_labels, batch_llms in val_loader:
            batch_labels = batch_labels.long().to('cuda:0')
            batch_llms = batch_llms.to('cuda:0').squeeze(1)
            outputs = model(batch_llms)
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == batch_labels).sum().item()
            val_total += batch_labels.size(0)
    
    # Test
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_graphs, batch_labels, batch_llms in test_loader:
            batch_labels = batch_labels.long().to('cuda:0')
            batch_llms = batch_llms.to('cuda:0').squeeze(1)
            outputs = model(batch_llms)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == batch_labels).sum().item()
            test_total += batch_labels.size(0)
    
    model.train()
    
    print(f"Epoch {epoch}: train_acc={train_correct/train_total:.3f}, "
          f"val_acc={val_correct/val_total:.3f}, test_acc={test_correct/test_total:.3f}, "
          f"train_loss={train_loss/len(train_loader):.4f}")

print("\n" + "=" * 70)
if test_correct / test_total >= 0.95:
    print("✅ LLM EMBEDDINGS ALONE CAN ACHIEVE >95% ACCURACY!")
    print("This is the REAL SOURCE of label leakage!")
else:
    print(f"❌ LLM embeddings alone achieve only {test_correct/test_total:.1%} accuracy")
    print(f"So problem is NOT in LLM embeddings - check graph features or data leakage")
