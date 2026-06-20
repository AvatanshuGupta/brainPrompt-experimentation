import torch
import os
import glob
import numpy as np
from torch.utils.data import DataLoader

from data.data import LoadData_llm
from train_TUs_graph_classification_llm import evaluate_network_all_metric

# ── Step 1: Find latest BrainPromptG ABIDE checkpoint run ─────────────────────
ckpt_base = 'out/braindata_graph_classification/checkpoints/'
pattern = os.path.join(ckpt_base, 'BrainPromptG_abide_full_AAL116_GPU0_*')

all_runs = glob.glob(pattern)

if not all_runs:
    print(f"No checkpoint directories found for pattern: {pattern}")
    exit()

latest_run = max(all_runs, key=os.path.getmtime)
print(f"Latest run: {latest_run}")

# ── Step 2: Load dataset exactly like training ────────────────────────────────
dataset = LoadData_llm(
    'abide_full_AAL116',
    threshold=0.3,
    edge_ratio=0,
    node_feat_transform='pearson'
)

device = torch.device('cpu')
fold_results = []

# ── Step 3: Evaluate best model of each fold ──────────────────────────────────
for fold in range(10):
    model_path = os.path.join(latest_run, f'RUN_{fold}', 'best_model.pkl')

    if not os.path.exists(model_path):
        print(f"Fold {fold}: model not found at {model_path}")
        continue

    print(f"\nEvaluating fold {fold} -> {model_path}")

    model = torch.load(model_path, map_location=device)
    model.eval()

    testset = dataset.test[fold]
    test_loader = DataLoader(
        testset,
        batch_size=34,
        shuffle=False,
        collate_fn=dataset.collate
    )

    _, test_acc, precision, recall, f1, roc_auc, _ = evaluate_network_all_metric(
        model, device, test_loader
    )

    fold_results.append({
        'fold': fold,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'path': model_path
    })

    print(f"Fold {fold}: "
          f"test_acc={test_acc:.4f}  "
          f"f1={f1:.4f}  "
          f"roc_auc={roc_auc:.4f}")

# ── Step 4: Find best fold ────────────────────────────────────────────────────
if fold_results:
    best = max(fold_results, key=lambda x: x['test_acc'])

    accs = [r['test_acc'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]
    aucs = [r['roc_auc'] for r in fold_results]

    print(f"\n{'='*60}")
    print(f"BEST FOLD:     {best['fold']}")
    print(f"BEST MODEL:    {best['path']}")
    print(f"Test Accuracy: {best['test_acc']:.4f}")
    print(f"F1 Score:      {best['f1']:.4f}")
    print(f"ROC-AUC:       {best['roc_auc']:.4f}")
    print("-"*60)
    print(f"AVG ACC across folds: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"AVG F1  across folds: {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}%")
    print(f"AVG AUC across folds: {np.mean(aucs)*100:.2f}% ± {np.std(aucs)*100:.2f}%")
    print(f"{'='*60}")

    import shutil
    best_save_path = 'best_model_overall.pkl'
    shutil.copy(best['path'], best_save_path)
    print(f"\n✓ Best model copied to: {best_save_path}")
else:
    print("No fold results found.")