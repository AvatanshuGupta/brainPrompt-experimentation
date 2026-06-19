# verify_fix.py — run after text2emb.py finishes
import torch
from dgl.data.utils import load_graphs

emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(emb)
unique = torch.unique(stacked, dim=0)
print("Total embeddings:", len(emb))
print("Unique embeddings:", unique.shape[0])          # should be ~1025

c0 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 1])
print("Between-class L2:", (c0.mean(0) - c1.mean(0)).norm().item())  # should be < 2
print("Within class-0 std:", c0.std(0).mean().item())                # should be > 0.1
print("Are first two identical?", torch.allclose(emb[0], emb[1]))    # should be False

# check_subject_order.py
import os
from dgl.data.utils import load_graphs

_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()
print("Total subjects in bin:", len(labels))
print("Label-0 count:", labels.count(0))
print("Label-1 count:", labels.count(1))

# Check folder names sorted — this is the order convert_abide_to_bin.py used
folders = sorted([
    f for f in os.listdir(r'D:\Datasets\abide')
    if os.path.isdir(os.path.join(r'D:\Datasets\abide', f))
])
print("\nTotal folders:", len(folders))
print("First 5:", folders[:5])
print("Last 5:", folders[-5:])

# Extract subject IDs from folder names
# sub-control50030 → id=50030, label=0
# sub-autism00001  → id=00001, label=1
for i, folder in enumerate(folders[:10]):
    inferred_label = 0 if 'control' in folder.lower() else 1
    print(f"  {folder} → label={inferred_label} | bin_label={labels[i]}")