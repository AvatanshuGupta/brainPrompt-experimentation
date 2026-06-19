"""Generate fixed (class-agnostic) label embeddings for ABIDE"""
import torch
from transformers import AutoTokenizer
from llm2vec.models import LlamaBiModel

print("Loading Llama-encoder-1.0B ...")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
model = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
model.eval()

device = torch.device("cpu")  # Use CPU only to avoid CUDA OOM
model = model.to(device)
print(f"Model loaded on: {device}")

# Read class-agnostic prompts
prompts = []
with open('data/prompts/label_prompts/abide_label_fixed.txt', 'r') as f:
    for line in f:
        prompts.append(line.strip())

print(f"\nFound {len(prompts)} prompts")
for i, p in enumerate(prompts):
    print(f"  [{i}]: {p[:80]}...")

# Encode each prompt
embeddings = []
for i, prompt in enumerate(prompts):
    print(f"\nEncoding prompt {i}...")
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden = outputs.last_hidden_state.squeeze()
    # For class-agnostic prompts, use all tokens (no special datatoken)
    emb = last_hidden.mean(dim=0)
    embeddings.append(emb.cpu())
    print(f"  Embedding shape: {emb.shape}, norm: {emb.norm():.2f}")

# Save embeddings
output_file = 'data/prompts/label_prompts/abide_label_fixed.pt'
torch.save(embeddings, output_file)
print(f"\n✅ Saved {len(embeddings)} embeddings to {output_file}")

# Verify embeddings are different from originals
print("\nVerifying embeddings differ from original class-specific ones...")
original = torch.load('data/prompts/label_prompts/abide_label.pt', weights_only=False)
original_stacked = torch.stack(original)

fixed_stacked = torch.stack(embeddings)

print(f"Original embeddings cosine similarity: {torch.nn.functional.cosine_similarity(original_stacked[0:1], original_stacked[1:2]).item():.4f}")
print(f"Fixed embeddings cosine similarity: {torch.nn.functional.cosine_similarity(fixed_stacked[0:1], fixed_stacked[1:2]).item():.4f}")
print(f"Original L2 distance: {torch.norm(original_stacked[0] - original_stacked[1]).item():.2f}")
print(f"Fixed L2 distance: {torch.norm(fixed_stacked[0] - fixed_stacked[1]).item():.2f}")
