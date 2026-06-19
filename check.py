import torch
emb = torch.load('data/prompts/meta_prompts/abide_meta_datatoken.pt')
print("Type:", type(emb))
print("Length:", len(emb))
print("Shape of first item:", emb[0].shape)
print("First item:", emb[0][:5])
print("Second item:", emb[1][:5])
print("Are first two identical?", torch.allclose(emb[0], emb[1]))

# Critical check — are embeddings identical within each class?
from data.data import LoadData_llm
from dgl.data.utils import load_graphs
_, Labels = load_graphs('brain_binfile/abide_full_AAL116.bin')
labels = Labels['glabel'].tolist()

class0 = [emb[i] for i, l in enumerate(labels) if l == 0]
class1 = [emb[i] for i, l in enumerate(labels) if l == 1]
c0 = torch.stack(class0)
c1 = torch.stack(class1)
print("\nWithin class-0 std:", c0.std(0).mean().item())
print("Within class-1 std:", c1.std(0).mean().item())
print("Between-class L2:", (c0.mean(0) - c1.mean(0)).norm().item())