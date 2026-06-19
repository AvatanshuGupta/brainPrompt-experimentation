import torch
from dgl.data.utils import load_graphs

emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(emb)

# Find which subjects share each unique embedding
unique, inverse = torch.unique(stacked, dim=0, return_inverse=True)
print(f"Unique embeddings: {unique.shape[0]}")

for uid in range(unique.shape[0]):
    mask = (inverse == uid).nonzero().squeeze().tolist()
    if isinstance(mask, int):
        mask = [mask]
    group_labels = [labels[i] for i in mask]
    n0 = group_labels.count(0)
    n1 = group_labels.count(1)
    print(f"Group {uid:2d}: {len(mask):3d} subjects | label-0: {n0:3d} | label-1: {n1:3d} | ratio: {n1/(n0+n1):.2f}")