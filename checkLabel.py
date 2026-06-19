# regenerate_abide_meta.py
import torch, csv
from sentence_transformers import SentenceTransformer  # or however your project encodes text

# Load the same encoder used in BrainPrompt
# Check your nets/ folder for how text encoding is done
# Common options:
model = SentenceTransformer('knowledgator/Llama-encoder-1.0B')  # from the paper

meta_path = 'data/prompts/meta_prompts/ABIDE_process.csv'
with open(meta_path, newline='') as f:
    rows = list(csv.reader(f))[1:]  # skip header

# Print first row so you can confirm column indices
print("Columns in row 0:", rows[0])

# Build subject-level prompts (adjust column indices after checking above)
prompts = []
for row in rows:
    # ADJUST THESE INDICES to match your actual CSV columns
    age        = row[2]   # e.g. "14"
    sex        = row[3]   # e.g. "male"
    site       = row[4]   # e.g. "NYU"
    total_sites = "20"    # hardcode or read from CSV

    prompt = (f"The subject is a(n) {age}-year-old {sex}, "
              f"with the image collected from the {site} site "
              f"out of a total of {total_sites} study locations.")
    prompts.append(prompt)

print(f"Total prompts: {len(prompts)}")
print("Example:", prompts[0])

# Encode
embeddings = model.encode(prompts, convert_to_tensor=True, show_progress_bar=True)
embedding_list = [embeddings[i] for i in range(len(embeddings))]

# Save — overwrites the corrupted file
torch.save(embedding_list, 'data/prompts/meta_prompts/abide_meta_datatoken.pt')
print("Done. Shape:", embedding_list[0].shape)

# Verify the fix
stacked = torch.stack(embedding_list)
print("\n=== VERIFICATION ===")
print("Std across all subjects:", stacked.std(0).mean().item())
print("Are first two identical?", torch.allclose(embedding_list[0], embedding_list[1]))
print("Expected: std > 0, identical = False")


import torch
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
stacked = torch.stack(emb)

# Reload labels
from dgl.data.utils import load_graphs
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

c0 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([emb[i] for i, l in enumerate(labels) if l == 1])

print("Within class-0 std:", c0.std(0).mean().item())   # should be >> 0
print("Within class-1 std:", c1.std(0).mean().item())   # should be >> 0  
print("Between-class L2:",  (c0.mean(0) - c1.mean(0)).norm().item())  # should be small
print("Are any two subjects identical?", torch.allclose(emb[0], emb[1]))  # should be False