# LongMemEval Setup Summary

## What We Accomplished ✅

You now have everything set up to run LongMemEval baselines with **temporal information as metadata** before implementing your temporal encoding experiments.

### 1. Data Ready
- **longmemeval_oracle.json** (15 MB): 500 questions with only evidence sessions
  - Quick to run, cheap to evaluate
  - 3-5 sessions per question
  - Perfect for initial baseline

- **longmemeval_s_cleaned.json** (265 MB): 500 questions with full history
  - ~40 sessions per question
  - ~115k tokens per question
  - Tests long-context temporal reasoning

### 2. Environment Configured
All Python dependencies installed:
- openai==1.35.1
- transformers, torch
- tqdm, backoff, numpy, nltk
- All evaluation tools ready

### 3. Analysis Tools Created

**analyze_temporal_metadata.py**
- Shows how dates currently appear in prompts
- Demonstrates the metadata approach (your baseline)
- Explains limitations you're trying to solve

**run_baseline.sh**
- One-command baseline execution
- Handles output directories automatically
- Easy to customize for different experiments

## How Temporal Metadata Currently Works

### In the Current Baseline (What You'll Compare Against)

Temporal information is **external metadata**:

```
### Session 1:
Session Date: 2023/04/10 (Mon) 14:47    ← METADATA (not part of conversation)
Session Content:
{"role": "user", "content": "I got my car serviced on March 15th..."}

Current Date: 2023/04/10 (Mon) 23:07     ← METADATA
Question: What was the first issue after the first service?
```

**Key characteristics:**
- Dates are labels, not content
- Model must parse date formats
- Temporal relationships are implicit
- Dense retrievers don't encode time (it's separate from text)

### Your Goal: Encode Time INTO Data

Instead of metadata, embed temporal information in the data itself:

**Option 1: Relative Time Tokens**
```
<T-5days> user: I got my car serviced...
```

**Option 2: Natural Language**
```
[5 days ago] user: I got my car serviced...
```

**Option 3: In Content**
```
user: Five days ago, I got my car serviced...
```

This allows:
- ✓ Dense retrievers to embed temporal similarity
- ✓ Model to learn temporal relationships directly
- ✓ More robust temporal reasoning
- ✓ Less reliance on prompt engineering

## Next Steps: Running Your Baseline

### Step 1: Set API Key
```bash
export OPENAI_API_KEY='your-openai-api-key'
```

### Step 2: Run Oracle Baseline (Recommended First)
```bash
cd benchmarks/LongMemEval/src/generation
bash run_generation.sh \
    ../../data/longmemeval_oracle.json \
    gpt-4o-mini \
    full-history-session \
    100 json false con
```

**This will:**
- Process 500 questions
- Cost ~$0.50-$1.00
- Take ~30-60 minutes
- Generate answers using temporal metadata (current approach)
- Save results to `generation_logs/`

### Step 3: Evaluate Results
```bash
cd benchmarks/LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o-mini <hypothesis-file> ../../data/longmemeval_oracle.json
```

### Step 4: Analyze Performance by Question Type

After evaluation, you'll get metrics for:
- **temporal-reasoning** (133 questions): Explicit temporal calculations
- **knowledge-update** (78 questions): Tracking changes over time
- **multi-session** (133 questions): Cross-session reasoning
- **single-session-user** (70 questions): Simple recall
- **single-session-preference** (30 questions): User preferences
- **single-session-assistant** (56 questions): Assistant recall

**Key metrics to track:**
- Overall accuracy with metadata approach
- Temporal reasoning question accuracy (most relevant to your work)
- Knowledge update accuracy (tracking time-based changes)

## Understanding the Codebase

### Key Files You Examined

1. **run_generation.py** ([link](benchmarks/LongMemEval/src/generation/run_generation.py:1))
   - Line 46-283: `prepare_prompt()` - Where temporal metadata is added
   - Line 252: Session date formatting
   - Line 280: Final prompt assembly with question date

2. **run_retrieval.py** ([link](benchmarks/LongMemEval/src/retrieval/run_retrieval.py:1))
   - Line 202-229: Corpus indexing (keeps timestamps separate)
   - Line 305-307: Timestamps stored parallel to text

3. **evaluate_qa.py** ([link](benchmarks/LongMemEval/src/evaluation/evaluate_qa.py:1))
   - LLM-as-judge evaluation
   - Different prompts for different question types
   - Handles temporal reasoning with "off-by-one" leniency

### Question Types Explained

From the paper and code analysis:

1. **single-session-user**: "What's my favorite restaurant?"
   - Recall user information from one session
   - Time mostly irrelevant

2. **single-session-preference**: "Plan a vacation for me"
   - Use user preferences appropriately
   - Time mostly irrelevant

3. **multi-session**: "What are my hobbies based on our conversations?"
   - Combine info from multiple sessions
   - Some temporal aspects (evolving interests)

4. **temporal-reasoning**: "What was the FIRST issue AFTER the service?"
   - **Explicit temporal reasoning required**
   - Must calculate temporal order/duration
   - **Most relevant for your temporal encoding work**

5. **knowledge-update**: "What was my best 5K time?" (improved over time)
   - Track changing information
   - Must use most recent value
   - **Also highly relevant for temporal encoding**

6. **abstention**: Questions with no answer in history
   - Model should say "I don't know"
   - Tests hallucination/grounding

## Baseline Results to Expect

Based on the paper, with metadata approach:

- **GPT-4o (oracle)**: ~70-80% accuracy
- **GPT-4o-mini (oracle)**: ~60-70% accuracy
- **Temporal reasoning**: Typically 10-15% lower than overall
- **Knowledge updates**: Similar to temporal reasoning

Your temporal encoding should aim to improve especially on:
- temporal-reasoning questions
- knowledge-update questions
- potentially retrieval performance (if encoding helps embeddings)

## File Organization

```
576-NLP-Final-Project/
├── benchmarks/LongMemEval/          # Submodule (original code)
│   ├── data/
│   │   ├── longmemeval_oracle.json          ✅ Downloaded
│   │   └── longmemeval_s_cleaned.json       ✅ Downloaded
│   └── src/
│       ├── generation/run_generation.py     📍 Generate answers
│       ├── retrieval/run_retrieval.py       📍 Run retrievers
│       └── evaluation/evaluate_qa.py        📍 Evaluate results
│
├── analyze_temporal_metadata.py     ✅ Your analysis script
├── run_baseline.sh                  ✅ Your baseline runner
├── BASELINE_GUIDE.md               ✅ Detailed instructions
├── SUMMARY.md                      ✅ This file
└── README.md                       ✅ Updated project README
```

## Research Workflow

### Phase 1: Baseline (Current - Ready to Run)
1. Run oracle baseline with temporal metadata
2. Analyze results by question type
3. Understand failure modes

### Phase 2: Temporal Encoding Design
1. Choose encoding approach (relative tokens, NL, embeddings, etc.)
2. Create data transformation scripts
3. Generate encoded versions of LongMemEval

### Phase 3: Evaluation
1. Run same experiments with encoded data
2. Compare to baseline (metadata approach)
3. Analyze improvements/regressions

### Phase 4: Analysis
1. Which question types improved?
2. Does encoding help retrieval?
3. Error analysis: new failure modes?

## Key Research Questions Answered by Baseline

1. **How accurate is temporal reasoning with metadata?**
   - Establishes upper bound for metadata approach
   - Shows where encoding might help

2. **Which temporal aspects are hardest?**
   - Duration calculation?
   - Temporal ordering?
   - Knowledge updates?

3. **Does prompt format matter?**
   - JSON vs natural language
   - Chain-of-thought vs direct

4. **How does context length affect temporal reasoning?**
   - Oracle (3-5 sessions) vs full history (40+ sessions)

## You're Ready! 🚀

Everything is set up to:
1. ✅ Run baseline experiments with temporal metadata
2. ✅ Analyze how temporal information currently works
3. ✅ Compare against your temporal encoding experiments

**Next command to run:**
```bash
export OPENAI_API_KEY='your-key'
./run_baseline.sh
```

Good luck with your temporal encoding research!
