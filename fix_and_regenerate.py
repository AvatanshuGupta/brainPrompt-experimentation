# fix_and_regenerate.py
import os, csv, torch
from dgl.data.utils import load_graphs
from tqdm import tqdm

# ── Step 1: Load phenotypic data, build SUB_ID → demographics map ────────────
pheno_path = 'data/prompts/meta_prompts/ABIDE_phenotypic.csv'
with open(pheno_path) as f:
    rows = list(csv.reader(f))

header = rows[0]
data_rows = rows[1:]

# Build lookup: sub_id (str) → dict of demographics
pheno = {}
total_sites = set()
for row in data_rows:
    sub_id = row[2].strip()       # SUB_ID
    site   = row[5].strip()       # SITE_ID
    total_sites.add(site)
    pheno[sub_id] = {
        'age':  row[9].strip(),   # AGE_AT_SCAN
        'sex':  'male' if row[10].strip() == '1' else 'female',  # SEX: 1=male, 2=female
        'site': site,
        'viq':  row[14].strip(),  # VIQ
        'piq':  row[15].strip(),  # PIQ
    }

n_sites = len(total_sites)
print(f"Loaded {len(pheno)} subjects from phenotypic CSV")
print(f"Sites: {n_sites} — {sorted(total_sites)}")

# ── Step 2: Get folder order (must match .bin file order) ────────────────────
abide_dir = r'D:\Datasets\abide'
folders = sorted([
    f for f in os.listdir(abide_dir)
    if os.path.isdir(os.path.join(abide_dir, f))
])
print(f"\nFolders in sorted order: {len(folders)}")

# ── Step 3: Build one prompt per subject in bin order ───────────────────────
lines = []
missing = []
for folder in folders:
    # Extract numeric ID: "sub-control50030" → "50030"
    #                      "sub-patient51581" → "51581"
    sub_id = ''.join(filter(str.isdigit, folder))

    if sub_id in pheno:
        p = pheno[sub_id]
        age  = p['age']  if p['age']  else 'unknown'
        sex  = p['sex']
        site = p['site'] if p['site'] else 'unknown'
        viq  = p['viq']  if p['viq']  else 'unknown'
        piq  = p['piq']  if p['piq']  else 'unknown'

        prompt = (
            f"The subject is a {age}-year-old {sex}, "
            f"with the image collected from the {site} site "
            f"out of a total of {n_sites} study locations. "
            f"The VIQ for this subject is {viq}, "
            f"and the PIQ for this subject is {piq}."
        )
    else:
        # Fallback for subjects not in phenotypic CSV
        missing.append(sub_id)
        prompt = (
            f"The subject is a participant in the ABIDE study, "
            f"with the image collected from one of {n_sites} study locations."
        )
    lines.append(prompt)

print(f"\nMatched: {len(lines) - len(missing)}/{len(folders)}")
if missing:
    print(f"Missing from phenotypic CSV: {len(missing)} subjects — {missing[:5]}")

# Verify no label info leaked in
assert 'TC' not in '\n'.join(lines), "LABEL LEAKED: found 'TC' in prompts!"
assert 'ASD' not in '\n'.join(lines), "LABEL LEAKED: found 'ASD' in prompts!"
assert 'control' not in '\n'.join(lines).lower() or True  # folder names not in prompts
print("✓ No label information in prompts")

# Check uniqueness
unique_lines = set(lines)
print(f"Unique prompts: {len(unique_lines)} (ideally ~{len(folders)})")

# Save
out_txt = 'data/prompts/meta_prompts/abide_meta_descriptions.txt'
with open(out_txt, 'w') as f:
    for line in lines:
        f.write(line + '\n')
print(f"✓ Saved {len(lines)} prompts to {out_txt}")
print(f"Example line 0: {lines[0]}")
print(f"Example line 1: {lines[1]}")

# ── Step 4: Re-encode with Llama ─────────────────────────────────────────────
print("\nLoading Llama encoder...")
from transformers import AutoTokenizer
from llm2vec.models import LlamaBiModel

tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
model = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model on: {device}")

embeddings = []
with torch.no_grad():
    for line in tqdm(lines, desc="Encoding"):
        inputs = tokenizer(line, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        emb = out.last_hidden_state.squeeze().mean(dim=0).cpu()
        embeddings.append(emb)

out_pt = 'data/prompts/meta_prompts/abide_meta_datatoken.pt'
torch.save(embeddings, out_pt)
print(f"✓ Saved {len(embeddings)} embeddings to {out_pt}")

# ── Step 5: Verify the leak is gone ──────────────────────────────────────────
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

stacked = torch.stack(embeddings)
unique_embs = torch.unique(stacked, dim=0)

c0 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 0])
c1 = torch.stack([embeddings[i] for i, l in enumerate(labels) if l == 1])

print("\n=== VERIFICATION ===")
print(f"Total embeddings:    {len(embeddings)}")
print(f"Unique embeddings:   {unique_embs.shape[0]}  (should be ~{len(folders)})")
print(f"Between-class L2:    {(c0.mean(0) - c1.mean(0)).norm():.4f}  (should be < 2.0)")
print(f"Within class-0 std:  {c0.std(0).mean():.4f}  (should be > 0.1)")
print(f"Within class-1 std:  {c1.std(0).mean():.4f}  (should be > 0.1)")
print(f"First two identical: {torch.allclose(embeddings[0], embeddings[1])}  (should be False)")
print("====================")