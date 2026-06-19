# debug_fast.py
print("Step 1: imports")
import torch
from tqdm import tqdm
print("Step 2: torch OK")

from sentence_transformers import SentenceTransformer
print("Step 3: SentenceTransformer imported")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Step 4: model loaded")

test = model.encode(["hello world", "test sentence"], convert_to_tensor=True)
print(f"Step 5: encoding works, shape={test.shape}")