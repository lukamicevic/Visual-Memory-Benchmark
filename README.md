# 576 NLP Final Project: Temporal Encoding for Long-Term Memory

**Goal:** Find a way to encode time into conversational data so models learn from it, instead of relying on temporal information as external metadata.

## Project Overview

This project explores temporal encoding methods for improving long-term memory in LLMs using the **LongMemEval benchmark**. Instead of presenting temporal information as metadata (e.g., `Session Date: 2023/05/20`), we aim to encode time directly into the data itself.

### Current Status: Baseline Setup Complete ✅

- ✅ LongMemEval benchmark data downloaded (oracle + S variants)
- ✅ Python environment configured with all dependencies
- ✅ Analysis tools created to understand current metadata approach
- ✅ Baseline experiment scripts ready to run

## Quick Start

### 1. Understand Current Approach (Metadata)

```bash
# See how temporal metadata currently appears in prompts
python3 analyze_temporal_metadata.py
```

This shows you:
- How dates are presented as metadata labels
- Where temporal information appears in prompts
- Limitations of the metadata approach

### 2. Run Baseline Experiments

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Run oracle baseline (quick, cheap)
cd benchmarks/LongMemEval/src/generation
bash run_generation.sh \
    ../../data/longmemeval_oracle.json \
    gpt-4o-mini \
    full-history-session \
    100 json false con
```

Or use the helper script:
```bash
./run_baseline.sh
```

### 3. Evaluate Results

```bash
cd benchmarks/LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o-mini <hypothesis-file> ../../data/longmemeval_oracle.json
```

## Repository Structure

```
576-NLP-Final-Project/
├── benchmarks/LongMemEval/              # Benchmark submodule
│   ├── data/
│   │   ├── longmemeval_oracle.json      # 500 questions, oracle retrieval
│   │   └── longmemeval_s_cleaned.json   # 500 questions, ~115k tokens each
│   └── src/                             # Generation, retrieval, evaluation
├── analyze_temporal_metadata.py         # Analyze current metadata usage
├── run_baseline.sh                      # Quick baseline runner
├── BASELINE_GUIDE.md                    # Detailed baseline guide
└── README.md                            # This file
```

## Next Steps: Temporal Encoding

After establishing baseline performance with metadata, experiment with encoding approaches:

1. **Relative Time Tokens**: `<T-5days>` instead of absolute dates
2. **Natural Language**: "5 days ago..." embedded in conversation
3. **Temporal Embeddings**: Learned time representations
4. **Event-based**: Temporal grounding as content

See [BASELINE_GUIDE.md](BASELINE_GUIDE.md) for detailed instructions.

## Research Questions

1. Does encoding time into data improve temporal reasoning accuracy?
2. Can temporal encoding help dense retrievers find relevant memories?
3. Which encoding method works best for different question types?
4. How does temporal encoding compare to metadata across different models?

## Documentation

- **[BASELINE_GUIDE.md](BASELINE_GUIDE.md)**: Complete guide to running baselines
- **[analyze_temporal_metadata.py](analyze_temporal_metadata.py)**: Script to understand current approach
- **[LongMemEval README](benchmarks/LongMemEval/README.md)**: Original benchmark documentation

## Branch: `temporal-encoding`

This branch contains experiments for encoding temporal information into conversational data for the LongMemEval benchmark.
