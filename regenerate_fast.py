# regenerate_fast.py
"""
Uses sentence-transformers (all-MiniLM-L6-v2) as a fast CPU-friendly encoder.
Produces 384-dim embeddings instead of 2048-dim.
NOTE: You will also need to update net_params['lm_dim'] in your config to 384.
"""
import torch
from tqdm import tqdm

with open('data/prompts/meta_prompts/abide_meta_descriptions.txt') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

print(f"Lines: {len(lines)}, Unique: {len(set(lines))}")
assert 'TC' not in '\n'.join(lines) and 'ASD' not in '\n'.join(lines)
print("✓ No label leakage")

# Install if needed: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB, very fast on CPU

print("Encoding...")
emb_matrix = model.encode(
    lines,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    device='cpu'
)  # shape: (1025, 384)

embeddings = [emb_matrix[i].cpu() for i in range(len(lines))]
torch.save(embeddings, 'data/prompts/meta_prompts/abide_meta_datatoken.pt')
print(f"✓ Saved. Shape: {embeddings[0].shape}")

# Verify
from dgl.data.utils import load_graphs
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(embeddings)
unique = torch.unique(stacked, dim=0)
c0 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 1])

print(f"\nUnique embeddings:  {unique.shape[0]}  (should be ~987)")
print(f"Between-class L2:   {(c0.mean(0) - c1.mean(0)).norm():.4f}  (should be < 2)")
print(f"Within-class std:   {c0.std(0).mean():.4f}  (should be > 0.1)")
print(f"First two same:     {torch.allclose(embeddings[0], embeddings[1])}  (should be False)")