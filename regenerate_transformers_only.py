# regenerate_transformers_only.py
# Uses only transformers + torch — no extra installs needed
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

print("Loading model (downloading ~90MB on first run)...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()
print("Model loaded")

with open('data/prompts/meta_prompts/abide_meta_descriptions.txt') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
print(f"Lines: {len(lines)}, Unique: {len(set(lines))}")

def encode_batch(texts, batch_size=64):
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        with torch.no_grad():
            out = model(**inputs)
        # Mean pool
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        embs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.extend([embs[j].cpu() for j in range(len(batch))])
    return all_embs

embeddings = encode_batch(lines)
torch.save(embeddings, 'data/prompts/meta_prompts/abide_meta_datatoken.pt')
print(f"✓ Saved {len(embeddings)} embeddings, dim={embeddings[0].shape[0]}")

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
print(f"\nNOTE: embedding dim is {embeddings[0].shape[0]} — update lm_dim in your config to match")