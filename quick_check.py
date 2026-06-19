# quick_check.py
import torch
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
stacked = torch.stack(emb)
unique = torch.unique(stacked, dim=0)
print("Unique embeddings:", unique.shape[0])
print("File size check - len:", len(emb))

# Also check the text file
with open('data/prompts/meta_prompts/abide_meta_descriptions.txt') as f:
    lines = f.readlines()
print("Lines in txt:", len(lines))
print("Unique lines:", len(set(lines)))
print("Line 0:", lines[0].strip())
print("Line 1:", lines[1].strip())
print("Contains 'TC':", any('TC' in l for l in lines))
print("Contains 'ASD':", any('ASD' in l for l in lines))