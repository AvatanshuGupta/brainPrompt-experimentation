# diagnose_remaining_leakage.py
import torch
import numpy as np
from dgl.data.utils import load_graphs
from data.data import LoadData_llm
from nets.load_net import gnn_model
from torch.utils.data import DataLoader

# Load dataset
dataset = LoadData_llm('abide_full_AAL116', threshold=0.3, node_feat_transform='pearson')
trainset = dataset.train[0]
testset  = dataset.test[0]

# ── Test 1: Verify meta embeddings are now clean ─────────────────────────────
print("=== TEST 1: Meta Embedding Check ===")
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
stacked = torch.stack(emb)
unique = torch.unique(stacked, dim=0)
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()
c0 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 1])
print(f"Unique embeddings: {unique.shape[0]}  (should be ~1025)")
print(f"Between-class L2:  {(c0.mean(0) - c1.mean(0)).norm():.4f}  (should be < 2)")
print(f"First two identical: {torch.allclose(emb[0], emb[1])}")

# ── Test 2: Check if node features (Pearson) separate classes perfectly ───────
print("\n=== TEST 2: Node Feature (Pearson) Separability ===")
all_feats, all_labels = [], []
for i in range(len(dataset.all)):
    g, label, llm = dataset.all[i]
    all_feats.append(g.ndata['feat'].mean().item())  # scalar summary per graph
    all_labels.append(int(label))

feats = np.array(all_feats)
labs  = np.array(all_labels)
c0_mean = feats[labs == 0].mean()
c1_mean = feats[labs == 1].mean()
c0_std  = feats[labs == 0].std()
c1_std  = feats[labs == 1].std()
print(f"Class-0 feat mean: {c0_mean:.4f} ± {c0_std:.4f}")
print(f"Class-1 feat mean: {c1_mean:.4f} ± {c1_std:.4f}")
print(f"Separation ratio:  {abs(c0_mean - c1_mean) / (c0_std + c1_std + 1e-8):.4f}")
print("If separation >> 1.0 → node features themselves may be leaking")

# ── Test 3: Check population graph construction in the model ─────────────────
print("\n=== TEST 3: Label Embeddings Check ===")
label_embs = torch.load('data/prompts/label_prompts/abide_label.pt')
label_embs = torch.stack(label_embs).squeeze()
print(f"Label embs shape: {label_embs.shape}")
# Check if any subject's llm embedding is close to a label embedding
close_to_label = 0
for i in range(len(emb)):
    d0 = (emb[i] - label_embs[0]).norm().item()
    d1 = (emb[i] - label_embs[1]).norm().item()
    if min(d0, d1) < 5.0:
        close_to_label += 1
print(f"Subject embeddings close to a label embedding: {close_to_label}/{len(emb)}")
print("Should be 0 if meta embeddings are clean")

# ── Test 4: Check train/test split for graph-level duplicates ────────────────
print("\n=== TEST 4: Train/Test Graph Overlap ===")
train_feats = set()
for g, y, llm in trainset:
    train_feats.add(hash(g.ndata['feat'].numpy().tobytes()))
test_feats = set()
for g, y, llm in testset:
    test_feats.add(hash(g.ndata['feat'].numpy().tobytes()))
overlap = train_feats & test_feats
print(f"Train graphs: {len(train_feats)}, Test graphs: {len(test_feats)}, Overlap: {len(overlap)}")

# ── Test 5: Can a linear classifier on node features alone get 100%? ─────────
print("\n=== TEST 5: Linear Separability of Node Features ===")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

X = np.array([dataset.all[i][0].ndata['feat'].numpy().flatten() 
              for i in range(len(dataset.all))])
y = np.array([int(dataset.all[i][1]) for i in range(len(dataset.all))])
print(f"Feature matrix shape: {X.shape}")
# Use small subset for speed
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
clf = LogisticRegression(max_iter=1000)
scores = cross_val_score(clf, X_pca, y, cv=5)
print(f"Logistic regression CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print("If this is > 0.95 → the Pearson node features are perfectly separable → leakage in feature construction")