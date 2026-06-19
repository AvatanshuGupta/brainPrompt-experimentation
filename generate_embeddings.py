"""
generate_embeddings.py
=======================
Generates all three .pt embedding files needed by BrainPrompt for ABIDE:

    1. data/prompts/label_prompts/abide_label.pt         (2 embeddings)
    2. data/prompts/ROI_prompts/AAL116_short_datatoken.pt (116 embeddings)
    3. data/prompts/meta_prompts/abide_meta_datatoken.pt  (N embeddings, one per subject)

This script uses the Llama-encoder (knowledgator/Llama-encoder-1.0B) 
which outputs 2048-dim embeddings — matching self.lm_dim = 2048 in the model.

Requirements:
    pip install transformers torch llm2vec

Usage:
    # Generate all embeddings at once:
    python generate_embeddings.py --all

    # Or generate individually:
    python generate_embeddings.py --labels
    python generate_embeddings.py --rois
    python generate_embeddings.py --meta --meta_txt data/prompts/meta_prompts/abide_meta_descriptions.txt
"""

import os
import torch
import argparse
from tqdm import tqdm


# ─────────────────────────────────────────────
# Load Llama encoder (done once, reused)
# ─────────────────────────────────────────────

def load_llama_encoder():
    """Load the Llama-encoder-1.0B model and tokenizer."""
    print("Loading Llama-encoder-1.0B ...")
    print("(First run will download ~2GB model from HuggingFace)")

    from transformers import AutoTokenizer
    from llm2vec.models import LlamaBiModel

    tokenizer = AutoTokenizer.from_pretrained("knowledgator/Llama-encoder-1.0B")
    model = LlamaBiModel.from_pretrained("knowledgator/Llama-encoder-1.0B")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    return model, tokenizer, device


# ─────────────────────────────────────────────
# Encode a single text string
# ─────────────────────────────────────────────

def encode_text(model, tokenizer, text: str, datatoken: str = '', device=None) -> torch.Tensor:
    """
    Encode one text string → 2048-dim embedding tensor (CPU).

    datatoken controls which token span to average:
      'label' → average tokens between start and the word 'subject'
      'ROI'   → average tokens from start to the colon ':'
      ''      → average all tokens
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt")

    if datatoken == 'label':
        begin_idx = 6
        # Find the token id for 'subject' (token id 4967 in LLaMA vocab)
        subject_positions = (inputs['input_ids'].squeeze() == 4967).nonzero(as_tuple=True)[0]
        if len(subject_positions) == 0:
            begin_idx, end_idx = 0, -1   # fallback: use all tokens
        else:
            end_idx = subject_positions[0].item()

    elif datatoken == 'ROI':
        begin_idx = 1
        # Find the colon token (token id 29901 in LLaMA vocab)
        colon_positions = (inputs['input_ids'].squeeze() == 29901).nonzero(as_tuple=True)[0]
        if len(colon_positions) == 0:
            begin_idx, end_idx = 0, -1
        else:
            end_idx = colon_positions[0].item()

    else:
        begin_idx, end_idx = 0, -1

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state.squeeze()   # (seq_len, 2048)

    if end_idx == -1:
        emb = last_hidden[begin_idx:].mean(dim=0)
    else:
        emb = last_hidden[begin_idx:end_idx].mean(dim=0)

    return emb.cpu()


# ─────────────────────────────────────────────
# Generate label embeddings
# ─────────────────────────────────────────────

def generate_label_embeddings(model, tokenizer, device,
                               txt_path='data/prompts/label_prompts/abide_label.txt',
                               out_path='data/prompts/label_prompts/abide_label.pt'):
    """
    Reads abide_label.txt (2 lines) and saves abide_label.pt.
    Each line → one 2048-dim embedding.
    Final .pt = list of 2 tensors, each shape (2048,)
    """
    print(f"\n[1/3] Generating LABEL embeddings from: {txt_path}")

    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"  Found {len(lines)} label descriptions")

    embeddings = []
    for line in tqdm(lines):
        emb = encode_text(model, tokenizer, line, datatoken='label', device=device)
        embeddings.append(emb)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"  ✓ Saved → {out_path}  (shape per embedding: {embeddings[0].shape})")


# ─────────────────────────────────────────────
# Generate ROI (brain region) embeddings
# ─────────────────────────────────────────────

def generate_roi_embeddings(model, tokenizer, device,
                             txt_path='data/prompts/ROI_prompts/AAL116_short.txt',
                             out_path='data/prompts/ROI_prompts/AAL116_short_datatoken.pt'):
    """
    Reads AAL116_short.txt (116 lines) and saves AAL116_short_datatoken.pt.
    Each line → one 2048-dim embedding.
    Final .pt = list of 116 tensors, each shape (2048,)
    """
    print(f"\n[2/3] Generating ROI embeddings from: {txt_path}")

    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"  Found {len(lines)} brain regions (expected 116)")

    embeddings = []
    for line in tqdm(lines):
        emb = encode_text(model, tokenizer, line, datatoken='ROI', device=device)
        embeddings.append(emb)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"  ✓ Saved → {out_path}  (shape per embedding: {embeddings[0].shape})")


# ─────────────────────────────────────────────
# Generate per-subject metadata embeddings
# ─────────────────────────────────────────────

def generate_meta_embeddings(model, tokenizer, device,
                              txt_path='data/prompts/meta_prompts/abide_meta_descriptions.txt',
                              out_path='data/prompts/meta_prompts/abide_meta_datatoken.pt'):
    """
    Reads the per-subject metadata text file (one line per subject, 
    in the SAME ORDER as the .bin file) and saves abide_meta_datatoken.pt.
    Final .pt = list of N tensors, each shape (2048,)
    """
    print(f"\n[3/3] Generating META embeddings from: {txt_path}")

    with open(txt_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"  Found {len(lines)} subjects")

    embeddings = []
    for line in tqdm(lines):
        emb = encode_text(model, tokenizer, line, datatoken='', device=device)
        embeddings.append(emb)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(embeddings, out_path)
    print(f"  ✓ Saved → {out_path}  (shape per embedding: {embeddings[0].shape})")


# ─────────────────────────────────────────────
# Verify saved embeddings
# ─────────────────────────────────────────────

def verify_embeddings():
    """Quick check that all three .pt files exist and have correct shapes."""
    print("\n" + "="*50)
    print("VERIFICATION")
    print("="*50)

    files = {
        'Label embeddings':
            'data/prompts/label_prompts/abide_label.pt',
        'ROI embeddings':
            'data/prompts/ROI_prompts/AAL116_short_datatoken.pt',
        'Meta embeddings':
            'data/prompts/meta_prompts/abide_meta_datatoken.pt',
    }

    all_ok = True
    for name, path in files.items():
        if not os.path.exists(path):
            print(f"  ✗ MISSING: {path}")
            all_ok = False
            continue

        embs = torch.load(path)
        n = len(embs)
        dim = embs[0].shape[0] if hasattr(embs[0], 'shape') else '?'
        print(f"  ✓ {name}: {n} embeddings, dim={dim}  [{path}]")

        if dim != 2048:
            print(f"    ⚠ WARNING: Expected dim=2048 (model lm_dim), got {dim}")
            all_ok = False

    if all_ok:
        print("\n✓ All embeddings look correct! You can now run training.")
    else:
        print("\n⚠ Some embeddings are missing or have wrong shape.")

    print("="*50)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LLM embeddings for BrainPrompt (ABIDE)')

    parser.add_argument('--all',    action='store_true', help='Generate all three embedding types')
    parser.add_argument('--labels', action='store_true', help='Generate label embeddings only')
    parser.add_argument('--rois',   action='store_true', help='Generate ROI embeddings only')
    parser.add_argument('--meta',   action='store_true', help='Generate metadata embeddings only')
    parser.add_argument('--verify', action='store_true', help='Verify existing .pt files')

    parser.add_argument('--meta_txt',
                        default='data/prompts/meta_prompts/abide_meta_descriptions.txt',
                        help='Path to the per-subject metadata text file '
                             '(output of convert_abide_to_bin.py)')

    args = parser.parse_args()

    if args.verify:
        verify_embeddings()
        exit(0)

    if not (args.all or args.labels or args.rois or args.meta):
        print("Please specify what to generate: --all, --labels, --rois, --meta, or --verify")
        parser.print_help()
        exit(1)

    # Load model once (shared across all generation tasks)
    model, tokenizer, device = load_llama_encoder()

    if args.all or args.labels:
        generate_label_embeddings(model, tokenizer, device)

    if args.all or args.rois:
        generate_roi_embeddings(model, tokenizer, device)

    if args.all or args.meta:
        generate_meta_embeddings(model, tokenizer, device, txt_path=args.meta_txt)

    verify_embeddings()
