# final_eval.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from data.data import LoadData_llm
from metrics import accuracy_TU as accuracy, precision, recall, f1, roc_auc, accuracy_all_classes
import torch.nn.functional as F

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = 'best_model_overall.pkl'
DATASET_NAME     = 'abide_full_AAL116'
THRESHOLD        = 0.3
NODE_FEAT        = 'pearson'
BATCH_SIZE       = 34
LABEL_PROMPT     = 'data/prompts/label_prompts/abide_label.pt'
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load dataset (full) ───────────────────────────────────────────────────────
print("Loading dataset...")
dataset = LoadData_llm(DATASET_NAME, threshold=THRESHOLD, node_feat_transform=NODE_FEAT)
full_loader = DataLoader(
    dataset.all,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=dataset.collate
)
print(f"Total subjects: {len(dataset.all)}")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from: {MODEL_PATH}")
model = torch.load(MODEL_PATH, map_location=DEVICE)
model = model.to(DEVICE)
print(f"Model architecture: {model.name}")
print(f"Model parameters:   {sum(p.numel() for p in model.parameters()):,}")
model.eval()

# Inject label embeddings (required by BrainPromptG forward pass)
label_embs = torch.stack(torch.load(LABEL_PROMPT)).squeeze()
model.label_embs = label_embs.to(DEVICE)

print(f"Model loaded on: {DEVICE}")

# ── Inference on full dataset ─────────────────────────────────────────────────
print("\nRunning inference on full dataset...")
all_preds  = []
all_probs  = []
all_labels = []
total_loss = 0.0

with torch.no_grad():
    for iter, (batch_graphs, batch_labels, batch_llms) in enumerate(full_loader):
        batch_graphs = batch_graphs.to(DEVICE)
        batch_x      = batch_graphs.ndata['feat'].to(DEVICE)
        batch_e      = batch_graphs.edata['feat'].to(DEVICE)
        batch_llms   = batch_llms.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE).long()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_llms)
        loss         = model.loss(batch_scores, batch_labels)

        total_loss  += loss.detach().item()
        all_preds.append(batch_scores.detach().cpu())
        all_probs.append(F.softmax(batch_scores, dim=1).detach().cpu())
        all_labels.append(batch_labels.detach().cpu())

# ── Aggregate ─────────────────────────────────────────────────────────────────
y_true  = torch.cat(all_labels, dim=0).numpy()
y_pred  = torch.cat(all_preds,  dim=0).numpy()
y_probs = torch.cat(all_probs,  dim=0).numpy()
avg_loss = total_loss / (iter + 1)

# ── Compute metrics ───────────────────────────────────────────────────────────
pred_classes = np.argmax(y_pred, axis=1)
correct      = (pred_classes == y_true).sum()
total        = len(y_true)

# Overall accuracy (manual, works for any n_classes)
overall_acc  = correct / total

# Binary metrics (ABIDE is binary: TC=0, ASD=1)
n_classes = len(np.unique(y_true))
if n_classes == 2:
    test_precision = precision(y_pred, y_true)
    test_recall    = recall(y_pred,    y_true)
    test_f1        = f1(y_pred,        y_true)
    test_roc_auc   = roc_auc(y_pred,   y_true)
else:
    # Multi-class fallback
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    test_precision = precision_score(y_true, pred_classes, average='macro')
    test_recall    = recall_score(   y_true, pred_classes, average='macro')
    test_f1        = f1_score(       y_true, pred_classes, average='macro')
    test_roc_auc   = roc_auc_score(  y_true, y_probs,      multi_class='ovr', average='macro')

# Per-class accuracy
all_acc = accuracy_all_classes(y_pred, y_true)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true, pred_classes)
class_names = ['TC (0)', 'ASD (1)'] if n_classes == 2 else [str(i) for i in range(n_classes)]

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL EVALUATION — FULL DATASET")
print("="*60)
print(f"  Total subjects evaluated : {total}")
print(f"  Correct predictions      : {correct}")
print(f"  Average loss             : {avg_loss:.4f}")
print()
print(f"  Accuracy   : {overall_acc * 100:.2f}%")
print(f"  Precision  : {test_precision * 100:.2f}%")
print(f"  Recall     : {test_recall * 100:.2f}%")
print(f"  F1 Score   : {test_f1 * 100:.2f}%")
print(f"  ROC-AUC    : {test_roc_auc * 100:.2f}%")
print()
print("  Per-class accuracy:")
for i, acc in enumerate(all_acc):
    print(f"    Class {i} ({class_names[i]}): {acc * 100:.2f}%")
print()
print("  Confusion Matrix:")
print(f"    Labels: {class_names}")
print(f"    {cm}")
print()
print("  Classification Report:")
print(classification_report(y_true, pred_classes, target_names=class_names))

# ── Prediction distribution ───────────────────────────────────────────────────
unique, counts = np.unique(pred_classes, return_counts=True)
print("  Prediction distribution:")
for cls, cnt in zip(unique, counts):
    print(f"    Class {cls}: {cnt} subjects ({cnt/total*100:.1f}%)")

print()
unique, counts = np.unique(y_true, return_counts=True)
print("  Ground truth distribution:")
for cls, cnt in zip(unique, counts):
    print(f"    Class {cls}: {cnt} subjects ({cnt/total*100:.1f}%)")

print("="*60)

# ── Save results to file ──────────────────────────────────────────────────────
results_path = 'final_eval_results.txt'
with open(results_path, 'w') as f:
    f.write("FINAL EVALUATION — FULL DATASET\n")
    f.write("="*60 + "\n")
    f.write(f"Model:             {MODEL_PATH}\n")
    f.write(f"Dataset:           {DATASET_NAME}\n")
    f.write(f"Total subjects:    {total}\n")
    f.write(f"Average loss:      {avg_loss:.4f}\n\n")
    f.write(f"Accuracy:          {overall_acc * 100:.2f}%\n")
    f.write(f"Precision:         {test_precision * 100:.2f}%\n")
    f.write(f"Recall:            {test_recall * 100:.2f}%\n")
    f.write(f"F1 Score:          {test_f1 * 100:.2f}%\n")
    f.write(f"ROC-AUC:           {test_roc_auc * 100:.2f}%\n\n")
    f.write("Per-class accuracy:\n")
    for i, acc in enumerate(all_acc):
        f.write(f"  Class {i} ({class_names[i]}): {acc * 100:.2f}%\n")
    f.write("\nConfusion Matrix:\n")
    f.write(f"{cm}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_true, pred_classes, target_names=class_names))

print(f"\n✓ Results saved to: {results_path}")