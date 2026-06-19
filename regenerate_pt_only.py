# regenerate_pt_only.py
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from llm2vec.models import LlamaBiModel

# Load the correct text file
with open('data/prompts/meta_prompts/abide_meta_descriptions.txt') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]

print(f"Lines to encode: {len(lines)}")
print(f"Unique lines: {len(set(lines))}")
print(f"Line 0: {lines[0]}")
print(f"Line 1: {lines[1]}")
assert 'TC' not in '\n'.join(lines) and 'ASD' not in '\n'.join(lines), "Label leaked!"
print("✓ No label leakage in text")

# Load model
print("\nLoading Llama encoder...")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
model = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
model.eval()

import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

device = torch.device("cpu")
model = model.to(device)

print(f"Running on: {device}")

# Encode
embeddings = []
with torch.no_grad():
    for line in tqdm(lines, desc="Encoding"):
        inputs = tokenizer(line, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        emb = out.last_hidden_state.squeeze().mean(dim=0).cpu()
        embeddings.append(emb)

# Save
out_path = 'data/prompts/meta_prompts/abide_meta_datatoken.pt'
torch.save(embeddings, out_path)
print(f"\n✓ Saved to {out_path}")

# Verify immediately
from dgl.data.utils import load_graphs
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(embeddings)
unique = torch.unique(stacked, dim=0)
c0 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 1])

print("\n=== FINAL VERIFICATION ===")
print(f"Unique embeddings:   {unique.shape[0]}   (should be ~987)")
print(f"Between-class L2:    {(c0.mean(0) - c1.mean(0)).norm():.4f}  (should be < 2.0)")
print(f"Within class-0 std:  {c0.std(0).mean():.4f}  (should be > 0.1)")
print(f"First two identical: {torch.allclose(embeddings[0], embeddings[1])}  (should be False)")
print("==========================")