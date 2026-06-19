import torch
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
from dgl.data.utils import load_graphs
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(emb)

# Check how many UNIQUE embeddings exist
unique = torch.unique(stacked, dim=0)
print("Total subjects:", len(emb))
print("Unique embeddings:", unique.shape[0])
# If this is 2 → exactly one embedding per class = pure label leakage
# If this is ~1025 → embeddings vary per subject (site/age differences)

# Check if subjects from same site have identical embeddings
print("\nFirst 5 embeddings identical to emb[0]:")
for i in range(20):
    if torch.allclose(emb[0], emb[i], atol=1e-4):
        print(f"  Subject {i}: label={labels[i]}")