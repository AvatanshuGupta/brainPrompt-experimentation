# regenerate_label_embs.py
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()

with open('data/prompts/label_prompts/abide_label.txt') as f:
    lines = [l.strip() for l in f.readlines() if l.strip()]
print("Label lines:", lines)

embeddings = []
with torch.no_grad():
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True, max_length=128)
        out = model(**inputs)
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        embeddings.append(emb.squeeze(0).cpu())

torch.save(embeddings, 'data/prompts/label_prompts/abide_label.pt')
print(f"Saved {len(embeddings)} label embeddings, dim={embeddings[0].shape[0]}")