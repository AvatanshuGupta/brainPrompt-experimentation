"""
convert_abide_to_bin.py
========================
Converts ABIDE per-subject .mat files into a single DGL binary graph file
(abide_full_aal116.bin) that BrainPrompt can load directly.

Expected input folder structure:
    abide/
        sub-control51364/
            sub-control51364_AAL116_features_timeseries.mat
            sub-control51364_AAL116_correlation_matrix.mat
            ...
        sub-autism00001/
            sub-autism00001_AAL116_features_timeseries.mat
            ...
        ...

Subject label is inferred from folder name:
    - "control" in name  → label 0  (TC = Typical Control)
    - "autism"  in name  → label 1  (ASD = Autism Spectrum Disorder)

Output:
    abide_full_aal116.bin   ← place this where BrainNet.py name2path points

Usage:
    python convert_abide_to_bin.py \
        --data_dir  /path/to/abide/ \
        --atlas     AAL116 \
        --out_file  /path/to/brain_binfile/abide_full_aal116.bin \
        --meta_out  data/prompts/meta_prompts/abide_meta_descriptions.txt
"""

import os
import glob
import argparse
import numpy as np
import torch
import dgl
import networkx as nx
from dgl.data.utils import save_graphs
from scipy.io import loadmat
from tqdm import tqdm


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_label_from_name(folder_name: str) -> int:
    """
    Infer class label from subject folder name.

    Label mapping:
        control  -> 0 (TC)
        autism   -> 1 (ASD)
        asd      -> 1 (ASD)
        patient  -> 1 (ASD)
    """

    name_lower = folder_name.lower()

    if "control" in name_lower:
        return 0

    elif (
        "autism" in name_lower
        or "asd" in name_lower
        or "patient" in name_lower
    ):
        return 1

    else:
        raise ValueError(
            f"Cannot infer label from folder name: '{folder_name}'\n"
            f"Folder must contain one of: "
            f"'control', 'autism', 'asd', or 'patient'."
        )


def load_mat_timeseries(mat_path: str) -> np.ndarray:
    """
    Load fMRI time-series from a .mat file.
    Returns array of shape (n_regions, n_timepoints).

    The code tries common variable names used in BrainDataset releases.
    If none match, it picks the largest numeric array in the file.
    """
    mat = loadmat(mat_path)

    # Common variable names (try in order)
    candidate_keys = [
        'ROISignals', 'roi_signals', 'timeseries', 'time_series',
        'features', 'data', 'X', 'tc'
    ]
    for key in candidate_keys:
        if key in mat and isinstance(mat[key], np.ndarray):
            arr = mat[key].astype(np.float32)
            # Some files store (T, N), others (N, T) — we want (N, T)
            if arr.ndim == 2:
                if arr.shape[0] < arr.shape[1]:   # already (N, T) since N=116 << T
                    return arr
                else:                              # stored as (T, N), transpose
                    return arr.T
            # 1-D or >2-D: fall through to fallback

    # Fallback: find the largest 2-D numeric array
    best = None
    best_size = 0
    for key, val in mat.items():
        if key.startswith('_'):
            continue
        if isinstance(val, np.ndarray) and val.ndim == 2:
            if val.size > best_size:
                best = val.astype(np.float32)
                best_size = val.size

    if best is None:
        raise ValueError(f"No 2-D array found in {mat_path}. Keys: {list(mat.keys())}")

    # Ensure shape (N, T) with N = n_regions (smaller dim)
    if best.shape[0] > best.shape[1]:
        best = best.T
    return best


# ─────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────

def convert(data_dir: str, atlas: str, out_file: str, meta_out: str):

    atlas_upper = atlas.upper()   # e.g. AAL116

    # ── 1. Collect all subject folders ──────────────────────────────────
    subject_folders = sorted([
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))
    ])

    if len(subject_folders) == 0:
        raise FileNotFoundError(
            f"No subfolders found in {data_dir}. "
            "Each subject must be in its own folder."
        )

    print(f"Found {len(subject_folders)} subject folders in: {data_dir}")

    graphs  = []
    labels  = []
    skipped = []
    meta_lines = []   # one text line per subject (for LLM embedding later)

    for folder_name in tqdm(subject_folders, desc="Converting subjects"):
        folder_path = os.path.join(data_dir, folder_name)

        # ── Infer label ──────────────────────────────────────────────────
        try:
            label = get_label_from_name(folder_name)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            skipped.append(folder_name)
            continue

        # ── Find the timeseries .mat file for this atlas ─────────────────
        # Pattern: sub-*_AAL116_features_timeseries.mat
        ts_pattern = os.path.join(folder_path, f'*{atlas_upper}*timeseries*.mat')
        ts_matches = glob.glob(ts_pattern)

        # Fallback: any .mat with the atlas name
        if not ts_matches:
            ts_pattern = os.path.join(folder_path, f'*{atlas_upper}*.mat')
            ts_matches = glob.glob(ts_pattern)
            # Prefer the features/timeseries file over correlation file
            ts_matches = [f for f in ts_matches if 'corr' not in f.lower()] or ts_matches

        if not ts_matches:
            print(f"  [SKIP] No {atlas_upper} timeseries .mat found in {folder_path}")
            skipped.append(folder_name)
            continue

        ts_path = ts_matches[0]

        # ── Load timeseries ───────────────────────────────────────────────
        try:
            ts = load_mat_timeseries(ts_path)   # shape: (n_regions, T)
        except Exception as e:
            print(f"  [SKIP] Failed to load {ts_path}: {e}")
            skipped.append(folder_name)
            continue

        n_regions, n_timepoints = ts.shape

        # ── Build DGL graph (fully connected directed) ────────────────────
        G_nx = nx.DiGraph(np.ones([n_regions, n_regions]))
        g = dgl.from_networkx(G_nx)

        # Node features: fMRI time series, shape (n_regions, T)
        g.ndata['N_features'] = torch.tensor(ts, dtype=torch.float32)

        # Edge features: placeholder (BrainNet.py overwrites with Pearson corr)
        g.edata['E_features'] = torch.ones(n_regions * n_regions, dtype=torch.float32)

        graphs.append(g)
        labels.append(label)

        # ── Build metadata text line ──────────────────────────────────────
        label_str = "TC" if label == 0 else "ASD"
        meta_lines.append(
            f"A brain fMRI scan of an ABIDE {label_str} subject "
            f"with {n_regions} brain ROIs and {n_timepoints} time points."
        )

    # ── 2. Summary ───────────────────────────────────────────────────────
    n_subjects = len(graphs)
    n_tc  = sum(1 for l in labels if l == 0)
    n_asd = sum(1 for l in labels if l == 1)

    print(f"\n{'='*50}")
    print(f"Subjects converted : {n_subjects}")
    print(f"  TC  (label=0)    : {n_tc}")
    print(f"  ASD (label=1)    : {n_asd}")
    print(f"Skipped            : {len(skipped)}")
    if skipped:
        print(f"  {skipped[:5]}{'...' if len(skipped)>5 else ''}")

    if n_subjects == 0:
        raise RuntimeError("No subjects were converted. Check your folder structure and labels.")

    # Quick shape check
    sample_shape = graphs[0].ndata['N_features'].shape
    print(f"Sample graph node feature shape: {sample_shape}  (n_regions={sample_shape[0]}, T={sample_shape[1]})")
    print(f"Sample label: {labels[0]}")

    # ── 3. Save DGL binary file ──────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    label_dict = {'glabel': torch.tensor(labels, dtype=torch.int64)}
    save_graphs(out_file, graphs, label_dict)
    print(f"\n✓ Saved DGL bin file → {out_file}")

    # ── 4. Save metadata text file (for LLM embedding) ───────────────────
    os.makedirs(os.path.dirname(os.path.abspath(meta_out)), exist_ok=True)
    with open(meta_out, 'w') as f:
        for line in meta_lines:
            f.write(line + '\n')
    print(f"✓ Saved metadata text → {meta_out}")
    print(f"  → Next step: run text2emb.py on this file to generate abide_meta_datatoken.pt")
    print(f"{'='*50}\n")

    return n_subjects


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ABIDE .mat files to DGL .bin')

    parser.add_argument('--data_dir',
                        required=True,
                        help='Root folder containing one subfolder per subject '
                             '(e.g., /Downloads/abide/)')

    parser.add_argument('--atlas',
                        default='AAL116',
                        choices=['AAL116', 'schaefer100', 'harvard48',
                                 'kmeans100', 'ward100'],
                        help='Atlas name to select the correct .mat files (default: AAL116)')

    parser.add_argument('--out_file',
                        default='brain_binfile/abide_full_aal116.bin',
                        help='Output path for the DGL .bin file')

    parser.add_argument('--meta_out',
                        default='data/prompts/meta_prompts/abide_meta_descriptions.txt',
                        help='Output path for per-subject metadata text '
                             '(feed this into text2emb.py)')

    args = parser.parse_args()

    convert(
        data_dir  = args.data_dir,
        atlas     = args.atlas,
        out_file  = args.out_file,
        meta_out  = args.meta_out,
    )
