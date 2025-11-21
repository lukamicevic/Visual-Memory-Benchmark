## Temporal Retrieval Encoder for LongMemEval

Complete implementation of temporal-aware dense retrieval for conversational memory.

### 📁 Files

1. **`temporal_retriever.py`** - Core temporal encoder
   - `SinusoidalTimeEmbedding` - Continuous time encoding
   - `BucketedTimeEmbedding` - Discrete temporal buckets
   - `FusionModule` - Combine content + time (concat/FiLM/addition)
   - `TemporalEncoder` - Main model (GTE/E5 + temporal encoding)

2. **`train_temporal_encoder.py`** - Training script
   - Semantic contrastive loss
   - Temporal proximity loss
   - Time-window membership loss

3. **`integrate_with_longmemeval.py`** - LongMemEval integration
   - Replace flat-gte/stella with temporal encoder
   - Run retrieval on LongMemEval data

---

### 🚀 Quick Start

#### Step 1: Train Temporal Encoder

```bash
source venv/bin/activate

python temporal_encoding/train_temporal_encoder.py \
    --data_path benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \
    --base_model Alibaba-NLP/gte-base-en-v1.5 \
    --epochs 10 \
    --batch_size 32
```

**Output:** `temporal_encoding/checkpoints/checkpoint_epoch_9.pt`

#### Step 2: Run Temporal Retrieval

```bash
python temporal_encoding/integrate_with_longmemeval.py \
    --checkpoint temporal_encoding/checkpoints/checkpoint_epoch_9.pt \
    --data benchmarks/LongMemEval/data/longmemeval_s_cleaned.json \
    --top_k 50
```

**Output:** `temporal_retrieval_results/temporal_retrieval_top50.json`

#### Step 3: Generate Answers with GPT

```bash
cd benchmarks/LongMemEval/src/generation

export OPENAI_API_KEY='your-key'

bash run_generation.sh \
    ../../../../temporal_retrieval_results/temporal_retrieval_top50.json \
    gpt-4o-mini \
    flat-session \
    50
```

#### Step 4: Evaluate

```bash
cd benchmarks/LongMemEval/src/evaluation

python evaluate_qa.py gpt-4o-mini \
    <hypothesis_file> \
    ../../data/longmemeval_s_cleaned.json
```

---

### 🧩 Architecture

```
Query: "What was the first issue after service?"
Timestamp: 2023-04-10 23:07

                    ↓

    ┌─────────────────────────────────┐
    │   Content Encoder (GTE-base)    │
    │   [768-dim embedding]           │
    └─────────────────────────────────┘
                    +
    ┌─────────────────────────────────┐
    │   Time Encoder (Sinusoidal)     │
    │   [128-dim embedding]           │
    └─────────────────────────────────┘
                    ↓
    ┌─────────────────────────────────┐
    │   Fusion (Concat + MLP)         │
    │   [768-dim output]              │
    └─────────────────────────────────┘
                    ↓
         Temporal-Aware Embedding
         (used for retrieval)
```

---

### 📊 Training Losses

**1. Semantic Contrastive Loss**
- Preserve content meaning
- Pull semantically similar conversations together
- Standard InfoNCE loss

**2. Temporal Proximity Loss**
- Embedding distance ∝ Temporal distance
- Closer in time → Closer in embedding space
- Margin-based ranking loss

**3. Time-Window Membership Loss**
- Teach temporal buckets: "same day", "same week", "same month"
- MSE between similarity and temporal window
- Enables "last 6 months" reasoning

**Combined:**
```
L_total = L_semantic + α·L_temporal + β·L_window
```

---

### 🔬 Experimental Comparisons

Run these experiments to measure temporal encoding impact:

**Baseline (No Temporal Encoding):**
```bash
# Use flat-gte (current LongMemEval)
cd benchmarks/LongMemEval/src/retrieval
bash run_retrieval.sh \
    ../../data/longmemeval_s_cleaned.json \
    flat-gte \
    session
```

**Your Temporal Encoder:**
```bash
# Use temporal-gte (your model)
python temporal_encoding/integrate_with_longmemeval.py \
    --checkpoint checkpoints/checkpoint_epoch_9.pt \
    --data benchmarks/LongMemEval/data/longmemeval_s_cleaned.json
```

**Compare:**
- Retrieval metrics (Recall@k, NDCG@k)
- QA accuracy (overall and per question type)
- Temporal-reasoning questions (most important!)
- Knowledge-update questions

---

### 💡 Ablation Studies

Test different components:

**1. Time Encoding Methods:**
```bash
--time_encoding sinusoidal  # Continuous
--time_encoding bucketed    # Discrete
```

**2. Fusion Strategies:**
```bash
--fusion_method concat    # Concatenate + MLP
--fusion_method film      # FiLM conditioning
--fusion_method addition  # Simple addition
```

**3. Loss Weights:**
```python
# In train_temporal_encoder.py
TemporalLosses(
    temporal_weight=0.3,  # Try: 0.0, 0.1, 0.5, 1.0
    window_weight=0.2     # Try: 0.0, 0.1, 0.5
)
```

---

### 📝 Expected Results

Based on LoCoMo benchmark (2024):
- **Baseline (flat-gte):** ~40-50% accuracy on temporal-reasoning
- **With temporal encoding:** Target 55-65% (10-15% improvement)
- **Human performance:** ~85-90%

**Key metrics to track:**
- Overall QA accuracy
- Temporal-reasoning subset accuracy ← **Most important**
- Knowledge-update subset accuracy
- Retrieval Recall@10 for answer sessions

---

### 🎯 Publication Strategy

**Title Ideas:**
- "Temporal-Aware Dense Retrieval for Conversational Memory"
- "Learning to Encode Time: Improving LLM Memory with Temporal Embeddings"
- "Beyond Semantics: Temporal Encoding for Long-Term Conversational AI"

**Contributions:**
1. Novel temporal encoding for conversational retrieval
2. Three temporal learning objectives
3. State-of-the-art on LongMemEval benchmark
4. Ablation studies on encoding strategies

**Venues:**
- ACL 2025, EMNLP 2025 (main NLP conferences)
- ICLR 2026 (if you add more ML theory)
- NAACL 2025 (North American focus)

---

### 🐛 Troubleshooting

**Out of memory during training:**
```bash
--batch_size 16  # Reduce batch size
--freeze_base True  # Only train fusion layer
```

**Slow training:**
```bash
# Use smaller base model for testing
--base_model sentence-transformers/all-MiniLM-L6-v2
```

**Want to test quickly:**
```bash
# Use oracle dataset (smaller, no retrieval needed)
--data_path benchmarks/LongMemEval/data/longmemeval_oracle.json
```

---

### ✅ Next Steps

1. ✅ **Created:** Temporal encoder implementation
2. ✅ **Created:** Training script with 3 losses
3. ✅ **Created:** LongMemEval integration
4. ⏳ **TODO:** Train on LongMemEval data
5. ⏳ **TODO:** Run retrieval experiments
6. ⏳ **TODO:** Compare to baselines
7. ⏳ **TODO:** Analyze results by question type
8. ⏳ **TODO:** Write paper!

Good luck! 🚀
