import torch
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
label_emb = torch.load('data/prompts/label_prompts/abide_label.pt')
label_embs = torch.stack(label_emb).squeeze()

print("label_embs shape:", label_embs.shape)  # should be [2, 2048]
print("Is subject 0 emb == label_embs[0]?", torch.allclose(emb[0], label_embs[0], atol=1e-4))
print("Is subject 0 emb == label_embs[1]?", torch.allclose(emb[0], label_embs[1], atol=1e-4))

# import csv
# with open('data/prompts/meta_prompts/ABIDE_process.csv') as f:
#     rows = list(csv.reader(f))
# print("Header:", rows[0])
# print("Row 1:", rows[1])
# print("Row 2:", rows[2])