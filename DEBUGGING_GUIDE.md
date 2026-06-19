# 🔍 BrainPrompt 100% Accuracy - Debugging Guide

## Problem Summary
Model achieving 100% accuracy on BOTH train and test = **Data Leakage** (not overfitting)

---

## ⚡ QUICK START: Run These Commands Now

### Step 1: Check if LLM embeddings are leaking labels
```powershell
python debug_llm_embeddings.py --dataset abide_full_AAL116
```

**What to look for:**
- If "Accuracy using ONLY LLM embeddings" > 0.95 → **FOUND THE LEAK**
- If < 0.80 → Leakage is elsewhere

---

### Step 2: Run training with enhanced diagnostics
```powershell
python llm-main.py --config configs/abide_full_AAL116/TUs_graph_classification_BrainPromptG_abide_full_AAL116_100k.json 2>&1 | tee debug_output.log
```

**New output to watch for:**

```
============================================================
EPOCH 0 - LLM EMBEDDING ANALYSIS (LEAKAGE TEST)
============================================================
Perfect separation by LLM alone: 100.0%
🚨 CRITICAL: LLM embeddings alone achieve >95% separation!
   This is YOUR DATA LEAK - embeddings encode the label!
```

And later:

```
============================================================
LEAKAGE DIAGNOSIS: Testing model component contribution
============================================================
Results on TEST set (103 samples):
  Combined (Graph + LLM):     1.0000
  Graph only (LLM=0):         0.5234  ← If this is random (0.5), then...
  LLM only (Graph feats=0):   0.9854  ← ...LLM is the problem!
```

---

## 🔎 Interpretation Guide

### Scenario 1: LLM-Only Perfect, Graph-Only Random
```
Combined: 0.99   Graph-Only: 0.52   LLM-Only: 0.98
🚨 DIAGNOSIS: LLM embeddings encode labels directly
```
**Root cause**: `data/prompts/label_prompts/` are class-specific  
**Fix**: Generate generic prompts that don't hint at the class

---

### Scenario 2: Both Good, Combined Perfect
```
Combined: 1.00   Graph-Only: 0.75   LLM-Only: 0.72
⚠️ DIAGNOSIS: Overfitting to training distribution
```
**Root cause**: Model learned spurious correlations  
**Fix**: 
- Increase regularization (dropout, weight decay)
- Reduce model capacity
- Expand dataset

---

### Scenario 3: All Components Perfect
```
Combined: 1.00   Graph-Only: 0.95   LLM-Only: 0.95
🚨 DIAGNOSIS: Multiple leakage sources or synthetic data
```
**Root cause**: Both components encode labels  
**Fix**: Audit entire data pipeline

---

## 📁 Where to Look for the Leak

### If LLM embeddings are the problem:

**Files to check:**
1. `data/prompts/label_prompts/abide_label.txt` 
   - Are descriptions different per class?
   - Example: "Control: healthy brain" vs "Disease: autism brain"
   
2. `generate_embeddings.py`
   - How are embeddings generated from prompts?
   - Are prompts being modified based on labels?

3. `data/data.py` (LoadData_llm)
   - How are batch_llms assigned to samples?
   - Is there any label-to-embedding mapping?

### If graph features are the problem:

**Files to check:**
1. `data/data.py` - Graph construction
2. `data/BrainNet.py` - Brain network processing
3. Check if correlation thresholding differs by class

---

## 🛠️ Debugging Checklist

### Pre-Training Diagnosis
```bash
# 1. Check embedding separability
python debug_llm_embeddings.py --dataset abide_full_AAL116

# Examine output:
# - "Perfect separation by LLM alone: X.X%"
# - Compare class mean embeddings
```

### During Training (Automatic)
Training now prints debug info at epochs 0, 1, 10, 50:
```
EPOCH 0 - LLM EMBEDDING ANALYSIS
- Class means comparison
- Perfect separation %
- Mean embedding difference
```

### After Training Completes
Automatic diagnosis runs:
```
[STEP 1] LEAKAGE DIAGNOSIS: Testing model components
[STEP 2] EVALUATION DIAGNOSTICS: Prediction confidence
```

---

## 🔧 How to Fix (Once Leak is Found)

### If LLM embeddings are leaking:

**Option 1: Use generic prompts**
```python
# Instead of class-specific descriptions:
# ❌ "Autism spectrum disorder: impaired social communication"
# ✓ "Person: brain network"
```

**Option 2: Randomize embeddings**
```python
# Add noise to break the class signal:
batch_llms = batch_llms + 0.1 * torch.randn_like(batch_llms)
```

**Option 3: Use separate train-test prompt generation**
```python
# Generate embeddings ONLY from graph structure
# Don't use label-related metadata at all
```

### If graph features are leaking:

**Check**: Is the correlation threshold applied differently per class?

```python
# ❌ BAD: threshold changes based on label
threshold = 0.3 if label == 0 else 0.2

# ✓ GOOD: consistent threshold for all
threshold = 0.3
```

---

## 📊 Expected Metrics After Fix

For a **balanced 2-class dataset**, you should see:
- Train accuracy: 70-90% (not 100%)
- Test accuracy: 65-85%
- Validation accuracy: ≈ Test accuracy (no overfitting)
- Per-class metrics:
  - Precision: 70-90%
  - Recall: 70-90%
  - F1: 70-90%

**Red flags:**
- Any class at 99%+ accuracy
- Test accuracy > 95% on medical data
- All metrics identical to train (overfitting)

---

## 📝 Files Modified for Debugging

1. **train_TUs_graph_classification_llm.py**
   - Added LLM embedding analysis in `train_epoch_sparse()` (epochs 0,1,10,50)
   - Added prediction diagnostics in `evaluate_network_all_metric()`
   - Added new `diagnose_data_leakage()` function

2. **llm-main.py**
   - Added call to `diagnose_data_leakage()` before final metrics

3. **NEW: debug_llm_embeddings.py**
   - Standalone script to check embeddings pre-training

---

## 🆘 Still Stuck?

Check this order:
1. Run `debug_llm_embeddings.py` - identifies the leak source
2. Look at Step 1 output ("Accuracy using ONLY LLM embeddings")
3. Check corresponding file from "Where to Look" section
4. Fix the root cause
5. Re-run training with diagnostics enabled

**If still getting 100% accuracy after fix:**
- The leak is from multiple sources
- Check both LLM embeddings AND graph structure
- Review `data/data.py` data loader carefully

