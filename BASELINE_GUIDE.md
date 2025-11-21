# LongMemEval Baseline Experiments Guide

This guide explains how to run baseline experiments on LongMemEval with **temporal information as metadata** (the current approach), before implementing your temporal encoding experiments.

## Setup Complete ✓

- [x] Downloaded LongMemEval data (oracle and S variants)
- [x] Installed Python dependencies
- [x] Analysis scripts created

## Understanding the Current Baseline

### How Temporal Metadata Works Now

Run this to see how dates appear in prompts:
```bash
python3 analyze_temporal_metadata.py
```

**Key findings:**
- Dates appear as **metadata labels**: `Session Date: 2023/04/10 (Mon) 14:47`
- Question date shown as: `Current Date: 2023/04/10 (Mon) 23:07`
- Temporal information is **external** to conversation content
- Model must parse date formats and calculate temporal relationships

### Prompt Structure Example

```
I will give you several history chats between you and a user. Please answer
the question based on the relevant chat history. Answer step by step.

History Chats:

### Session 1:
Session Date: 2023/04/10 (Mon) 14:47   ← METADATA
Session Content:
[{"role": "user", "content": "..."}, ...]

Current Date: 2023/04/10 (Mon) 23:07    ← METADATA
Question: What was the first issue I had with my new car after its first service?
Answer (step by step):
```

## Running Baseline Experiments

### Option 1: Quick Start (Oracle Setting)

The oracle setting only includes evidence sessions - easiest way to test temporal reasoning:

```bash
# Set your API key
export OPENAI_API_KEY='your-openai-key-here'

# Run oracle baseline (only 3-5 sessions per question)
cd benchmarks/LongMemEval/src/generation
bash run_generation.sh \
    ../../data/longmemeval_oracle.json \
    gpt-4o-mini \
    full-history-session \
    100 \
    json \
    false \
    con
```

**What this tests:**
- How well does the model use temporal metadata for reasoning?
- Can it track knowledge updates over time?
- Does it handle temporal-reasoning questions correctly?

### Option 2: Full LongMemEval_S (Long Context)

Tests with ~40 sessions (115k tokens):

```bash
cd benchmarks/LongMemEval/src/generation
bash run_generation.sh \
    ../../data/longmemeval_s_cleaned.json \
    gpt-4o-mini \
    full-history-session \
    1000 \
    json \
    false \
    con
```

**Note:** Requires model with 128k+ context window

### Option 3: Using Our Helper Script

```bash
# Make executable
chmod +x run_baseline.sh

# Run with defaults (oracle, gpt-4o-mini)
./run_baseline.sh

# Custom configuration
./run_baseline.sh data/longmemeval_s_cleaned.json gpt-4o-mini full-history-session 1000
```

## Evaluation

After generation completes, evaluate the results:

```bash
cd benchmarks/LongMemEval/src/evaluation

# Evaluate with GPT-4o-mini as judge
python evaluate_qa.py gpt-4o-mini \
    <path-to-hypothesis-file> \
    ../../data/longmemeval_oracle.json

# View aggregated metrics
python print_qa_metrics.py gpt-4o-mini \
    <hypothesis-file>.eval-results-gpt-4o-mini \
    ../../data/longmemeval_oracle.json
```

## What to Analyze

### 1. Overall Performance
- **Overall Accuracy**: How well does model answer with temporal metadata?
- **Per-question-type**: Which types benefit from temporal info?
  - `temporal-reasoning`: Explicitly requires temporal calculation
  - `knowledge-update`: Must track changing facts over time
  - `multi-session`: Cross-session reasoning
  - `single-session-*`: Simpler recall tasks

### 2. Temporal Metadata Usage
Look at generated answers to see:
- Does model correctly parse dates? (e.g., "March 15th" vs "2023/03/15")
- Can it calculate durations? ("15 days after")
- Does it track temporal order? ("first issue after service")

### 3. Common Failure Modes
- **Date format confusion**: Model struggles with date parsing
- **Temporal reasoning errors**: Wrong calculations (off-by-one, duration)
- **Knowledge update conflicts**: Using old info instead of updated
- **Metadata hallucination**: Inventing dates not in context

## Next Steps: Temporal Encoding Experiments

After establishing baseline performance, you can experiment with:

### Encoding Approaches to Test

1. **Relative Time Tokens**
   - Replace `Session Date: 2023/04/10` with `<T-5days>` (5 days before question)
   - Model learns temporal distances directly

2. **Natural Language Encoding**
   - "This conversation happened 5 days ago..."
   - More human-readable, easier to reason over

3. **Temporal Embeddings**
   - Add learned time embeddings to conversation tokens
   - Continuous time representation

4. **Event-based Encoding**
   - "After the first car service..." in conversation itself
   - Make temporal grounding part of content

### Comparison Framework

For each encoding method, measure:
- **QA Accuracy** vs baseline (with metadata)
- **Retrieval Performance**: Does encoding help dense retrievers?
- **Temporal Question Accuracy**: Focus on temporal-reasoning subset
- **Knowledge Update Accuracy**: Can it track changes better?

### Hypothesis to Test

**Main hypothesis:** Encoding time INTO data will:
- ✓ Improve dense retrieval (time encoded in embeddings)
- ✓ Reduce temporal reasoning errors (explicit relationships)
- ✓ Better knowledge update tracking (time as content, not metadata)
- ✓ More robust across models (less prompt engineering needed)

## Directory Structure

```
576-NLP-Final-Project/
├── benchmarks/LongMemEval/          # Submodule
│   ├── data/
│   │   ├── longmemeval_oracle.json         # ✓ Downloaded
│   │   └── longmemeval_s_cleaned.json      # ✓ Downloaded
│   └── src/
│       ├── generation/              # Generate answers
│       ├── retrieval/               # Run retrievers
│       └── evaluation/              # Evaluate QA
├── baseline_results/                # Your baseline runs go here
├── temporal_encoding_experiments/   # Your encoding experiments
├── analyze_temporal_metadata.py     # Analysis script
├── run_baseline.sh                  # Quick baseline runner
└── BASELINE_GUIDE.md               # This file
```

## Estimated Costs (GPT-4o-mini)

- **Oracle (500 questions)**: ~$0.50 - $1.00
  - Small context (3-5 sessions)
  - Quick baseline

- **LongMemEval_S (500 questions)**: ~$5 - $10
  - Large context (~115k tokens/question)
  - Full evaluation

## Questions?

Key things established:
1. ✅ Data downloaded and ready
2. ✅ Dependencies installed
3. ✅ Scripts created for baseline
4. ✅ Understanding of how temporal metadata currently works

You're ready to run baselines and then experiment with temporal encoding!
