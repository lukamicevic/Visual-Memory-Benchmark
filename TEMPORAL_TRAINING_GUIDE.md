# Temporal Encoder Training Guide

Complete step-by-step guide for training and evaluating the temporal encoder.

---

## Overview

**Goal:** Train a temporal-aware retrieval encoder that embeds timestamp information directly into the vector space, instead of using temporal metadata as text.

**Key Difference:**
- **Baseline:** `Session Date: 2023/04/10 (Mon) 14:47` as text metadata
- **Temporal Encoder:** Timestamp embedded as `sin/cos` encoding → fused with content embedding → model learns temporal relationships

---

## Step 1: Environment Setup

### Check Dependencies

```bash
cd /scratch/lmicevic/Visual-Memory-Benchmark

# Check if packages are installed
python3 -c "import torch; import transformers; import sentence_transformers; print('✅ All packages available')"
```

### Install Missing Packages (if needed)

```bash
pip3 install --user torch transformers sentence-transformers tqdm numpy
```

---

## Step 2: Train Temporal Encoder on Oracle Dataset

### Quick Training (Oracle Dataset - Recommended First)

The oracle dataset is smaller and faster for initial testing.

```bash
cd /scratch/lmicevic/Visual-Memory-Benchmark

# Create checkpoint directory
mkdir -p temporal_encoding/checkpoints

# Train on oracle dataset (500 questions, 3-5 sessions each)
python3 temporal_encoding/train_temporal_encoder.py \
    --data_path benchmarks/LongMemEval/data/longmemeval_oracle.json \
    --base_model Alibaba-NLP/gte-base-en-v1.5 \
    --epochs 5 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --time_encoding sinusoidal \
    --fusion_method concat \
    --temporal_weight 0.3 \
    --window_weight 0.2 \
    --save_dir temporal_encoding/checkpoints
```

**Expected Output:**
```
Loaded 1500-2000 sessions from 500 questions
Epoch 1/5: 100%|████████| Loss: 0.XXX
Epoch 2/5: 100%|████████| Loss: 0.XXX
...
✅ Checkpoint saved: temporal_encoding/checkpoints/checkpoint_epoch_4.pt
```

**Training Time:** ~10-30 minutes (depending on GPU availability)

---

## Step 3: Verify Training Success

```bash
# Check if checkpoints were created
ls -lh temporal_encoding/checkpoints/

# Expected output:
# checkpoint_epoch_0.pt
# checkpoint_epoch_1.pt
# ...
# checkpoint_epoch_4.pt
```

---

## Step 4: Run Temporal Retrieval

Use the trained encoder to retrieve sessions with temporal awareness.

```bash
cd /scratch/lmicevic/Visual-Memory-Benchmark

# Create output directory
mkdir -p temporal_retrieval_results

# Run temporal retrieval on oracle dataset
python3 temporal_encoding/integrate_with_longmemeval.py \
    --checkpoint temporal_encoding/checkpoints/checkpoint_epoch_4.pt \
    --data benchmarks/LongMemEval/data/longmemeval_oracle.json \
    --top_k 100 \
    --output temporal_retrieval_results/temporal_oracle_top100.json
```

**Expected Output:**
```
Loading checkpoint: temporal_encoding/checkpoints/checkpoint_epoch_4.pt
✅ Temporal encoder loaded
Encoding corpus: 100%|████████| 1500/1500 sessions
Running retrieval: 100%|████████| 500/500 questions
✅ Results saved: temporal_retrieval_results/temporal_oracle_top100.json
```

---

## Step 5: Generate Answers with Temporal Retrievals

Now use GPT to generate answers based on temporal-encoded retrievals.

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

cd benchmarks/LongMemEval/src/generation

# Generate answers using temporal retrievals
python3 run_generation.py \
    --in_file ../../../../temporal_retrieval_results/temporal_oracle_top100.json \
    --out_dir ../../../../temporal_results \
    --model_name gpt-4o-mini-2024-07-18 \
    --model_alias gpt-4o-mini \
    --retriever_type oracle-session \
    --openai_key "$OPENAI_API_KEY" \
    --topk_context 100 \
    --history_format json \
    --useronly false \
    --cot true
```

**Cost:** ~$0.50-$1.00 for oracle dataset

---

## Step 6: Evaluate Results

Compare temporal encoder vs baseline (metadata approach).

```bash
cd benchmarks/LongMemEval/src/evaluation

# Evaluate temporal encoder results
python3 evaluate_qa.py gpt-4o-mini \
    <temporal_hypothesis_file> \
    ../../data/longmemeval_oracle.json

# Evaluate baseline results (if you ran baseline)
python3 evaluate_qa.py gpt-4o-mini \
    <baseline_hypothesis_file> \
    ../../data/longmemeval_oracle.json
```

### View Detailed Metrics

```bash
# Print aggregated metrics
python3 print_qa_metrics.py gpt-4o-mini \
    <temporal_hypothesis_file>.eval-results-gpt-4o-mini \
    ../../data/longmemeval_oracle.json
```

**Key Metrics to Compare:**
- Overall accuracy
- **Temporal-reasoning accuracy** ← Most important!
- Knowledge-update accuracy
- Multi-session accuracy

---

## Step 7: Scaling to Full Dataset (Optional)

If oracle results look promising, train on the full S dataset.

```bash
cd /scratch/lmicevic/Visual-Memory-Benchmark

# Train on full S dataset (~40 sessions per question)
python3 temporal_encoding/train_temporal_encoder.py \
    --data_path benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \
    --base_model Alibaba-NLP/gte-base-en-v1.5 \
    --epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --time_encoding sinusoidal \
    --fusion_method concat \
    --temporal_weight 0.3 \
    --window_weight 0.2 \
    --save_dir temporal_encoding/checkpoints_full
```

**Training Time:** ~1-3 hours (depending on hardware)

---

## Training Parameters Explained

### Architecture Parameters

**`--base_model`** (default: `Alibaba-NLP/gte-base-en-v1.5`)
- Pre-trained sentence encoder for content
- Alternatives: `sentence-transformers/all-MiniLM-L6-v2` (faster, smaller)

**`--time_encoding`** (default: `sinusoidal`)
- `sinusoidal`: Continuous time encoding (sin/cos functions)
- `bucketed`: Discrete temporal buckets (day, week, month, etc.)

**`--fusion_method`** (default: `concat`)
- `concat`: Concatenate content + time, then MLP projection
- `film`: FiLM conditioning (scale/shift content with time)
- `addition`: Simple element-wise addition

### Loss Weights

**`--temporal_weight`** (default: `0.3`)
- Weight for temporal proximity loss
- Higher = stronger emphasis on time similarity
- Try: 0.0, 0.1, 0.5, 1.0

**`--window_weight`** (default: `0.2`)
- Weight for time-window membership loss
- Teaches "same day", "same week", etc.
- Try: 0.0, 0.1, 0.5

### Training Hyperparameters

**`--epochs`** (default: `10`)
- Number of training epochs
- Oracle: 5-10 epochs sufficient
- Full dataset: 10-20 epochs

**`--batch_size`** (default: `16`)
- Reduce if out of memory (try 8 or 4)
- Increase if you have GPU (try 32 or 64)

**`--learning_rate`** (default: `2e-5`)
- Standard fine-tuning rate
- Try: 1e-5 (slower), 5e-5 (faster)

---

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
--batch_size 4

# Use smaller base model
--base_model sentence-transformers/all-MiniLM-L6-v2
```

### Training Too Slow

```bash
# Use smaller dataset first
--data_path benchmarks/LongMemEval/data/longmemeval_oracle.json

# Reduce max sessions
--max_sessions 10
```

### Module Not Found Error

```bash
# Install missing packages
pip3 install --user torch transformers sentence-transformers tqdm

# Or use requirements.txt
pip3 install --user -r requirements.txt
```

---

## Expected Results

Based on similar benchmarks (LoCoMo, 2024):

**Baseline (Metadata Approach):**
- Overall accuracy: ~60-70%
- Temporal-reasoning: ~40-50%

**Temporal Encoder (Target):**
- Overall accuracy: ~65-75% (+5% improvement)
- Temporal-reasoning: ~55-65% (+10-15% improvement)

**Human Performance:**
- Overall accuracy: ~85-90%

---

## Ablation Studies (Advanced)

Test different components to understand what helps:

### 1. Time Encoding Method

```bash
# Sinusoidal (continuous)
--time_encoding sinusoidal

# Bucketed (discrete)
--time_encoding bucketed
```

### 2. Fusion Strategy

```bash
# Concatenation (baseline)
--fusion_method concat

# FiLM conditioning
--fusion_method film

# Addition
--fusion_method addition
```

### 3. Loss Weights

```bash
# Semantic only (no temporal loss)
--temporal_weight 0.0 --window_weight 0.0

# Balanced
--temporal_weight 0.3 --window_weight 0.2

# High temporal emphasis
--temporal_weight 1.0 --window_weight 0.5
```

---

## Quick Start Summary

```bash
# 1. Train temporal encoder (oracle dataset)
python3 temporal_encoding/train_temporal_encoder.py \
    --data_path benchmarks/LongMemEval/data/longmemeval_oracle.json \
    --epochs 5 --batch_size 16

# 2. Run temporal retrieval
python3 temporal_encoding/integrate_with_longmemeval.py \
    --checkpoint temporal_encoding/checkpoints/checkpoint_epoch_4.pt \
    --data benchmarks/LongMemEval/data/longmemeval_oracle.json

# 3. Generate answers (with OpenAI API key)
export OPENAI_API_KEY='your-key'
cd benchmarks/LongMemEval/src/generation
python3 run_generation.py --in_file ../../../../temporal_retrieval_results/...

# 4. Evaluate
cd ../evaluation
python3 evaluate_qa.py gpt-4o-mini <hypothesis_file> ../../data/longmemeval_oracle.json
```

---

## Next Steps After Training

1. **Compare Results:** Baseline vs Temporal Encoder
   - Focus on temporal-reasoning questions
   - Analyze failure cases

2. **Error Analysis:**
   - Which question types improved most?
   - Where does temporal encoding still fail?

3. **Write Paper:** Report findings!
   - Novel temporal encoding approach
   - State-of-the-art on LongMemEval benchmark
   - Ablation studies on encoding strategies

---

Good luck! 🚀
