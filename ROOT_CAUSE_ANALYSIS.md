# BrainPrompt 100% Accuracy - ROOT CAUSE ANALYSIS

## Problem Summary
Model achieves 100% accuracy on both train and test sets for ABIDE autism classification (2-class: TC vs ASD).

## Root Cause: CLASS-SPECIFIC LABEL PROMPTS

### Discovery Process
1. Disabled auxiliary loss in BrainPromptG & BrainPromptC - **100% accuracy persisted**
2. Tested LLM embeddings alone with simple 3-layer MLP classifier:
   - **Result: 100% accuracy by Epoch 1!**
   - This proves LLM embeddings themselves encode perfect label signal
3. Examined label prompt generation code
   - **Found:** Label prompts explicitly mention class labels:
     ```
     Label 0: "A brain network of a TC subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD."
     Label 1: "A brain network of a ASD subject. The entire population is categorized into two groups: typical controls (TC) and individuals diagnosed with ASD."
     ```

### Why This Causes 100% Accuracy
- BERT encoder reads "TC subject" vs "ASD subject" 
- Generates semantically distinct 2048-dim embeddings
- These embeddings ARE the class labels - perfectly separable!
- Model fusion of graph features + these class-specific embeddings → trivial classification

## Evidence
- LLM-only classifier accuracy over 5 epochs:
  ```
  Epoch 0: train_acc=0.873, val_acc=1.000, test_acc=1.000
  Epoch 1: train_acc=1.000, val_acc=1.000, test_acc=1.000  ← PERFECT!
  Epoch 2: train_acc=1.000, val_acc=1.000, test_acc=1.000
  Epoch 3: train_acc=1.000, val_acc=1.000, test_acc=1.000
  Epoch 4: train_acc=1.000, val_acc=1.000, test_acc=1.000
  ```

## The Fix

### Option 1: Use Class-Agnostic Label Prompts (RECOMMENDED)
Replace class-specific prompts with class-agnostic ones:

**Current (WRONG):**
```
Label 0: "A brain network of a TC subject..."
Label 1: "A brain network of a ASD subject..."
```

**Fixed (CORRECT):**
```
Label 0: "A brain network sample representing one classification category in autism research."
Label 1: "A brain network sample representing the other classification category in autism research."
```

Or even simpler:
```
Label 0: "Category A brain network"
Label 1: "Category B brain network"
```

### Option 2: Generate New Label Embeddings
Create new prompt files:
- `data/prompts/label_prompts/abide_label_fixed.txt` - with class-agnostic descriptions
- `data/prompts/label_prompts/abide_label_fixed.pt` - corresponding BERT embeddings
- Update config to use new file

### Option 3: Disable Label Embeddings Entirely (QUICK TEST)
Set `lambda1 = 0` in config and verify:
- Graph features alone should give ~50-70% accuracy (weak signal)
- Confirms label embeddings are the only strong signal

## Implementation Steps

1. **Edit label prompts:**
   ```bash
   # File: data/prompts/label_prompts/abide_label.txt
   # Replace with class-agnostic descriptions
   ```

2. **Regenerate embeddings:**
   ```bash
   python generate_embeddings.py --input data/prompts/label_prompts/abide_label.txt \
                                  --output data/prompts/label_prompts/abide_label.pt \
                                  --model llama
   ```

3. **Retrain and verify:**
   ```bash
   venv\Scripts\python llm-main.py --config configs/abide_full_AAL116/TUs_graph_classification_BrainPromptG_abide_full_AAL116_100k.json --epochs 100
   ```
   
   **Expected result:** 
   - Epoch 0: train_acc ≈ 50-60%, test_acc ≈ 50-60%
   - Epoch 10: train_acc ≈ 70-80%, test_acc ≈ 65-75%
   - No more 100% accuracy

## Files to Modify

1. `data/prompts/label_prompts/abide_label.txt` - Replace with class-agnostic prompts
2. `data/prompts/label_prompts/abide_label.pt` - Regenerate embeddings
3. Similarly for `adni_label.txt` if applicable

## Verification

Run the LLM-only classifier test again after fix:
```bash
venv\Scripts\python test_llm_only_classifier.py
```

**Expected after fix:**
- Should NOT achieve 100% accuracy
- Should get random ~50% accuracy with simple MLP
- Proves label embeddings no longer encode class identity
