# Quick Start Guide - Virtual Environment Ready! 🚀

## ✅ Setup Complete

Your virtual environment is created and all dependencies are installed!

## Run Your Baseline Now

### Step 1: Get Your OpenAI API Key

Go to: https://platform.openai.com/api-keys
- Sign up or log in
- Create a new API key
- Copy it (starts with `sk-proj-...`)

### Step 2: Set Your API Key

```bash
export OPENAI_API_KEY='sk-proj-your-actual-key-here'
```

### Step 3: Run the Baseline

```bash
cd /Users/lukamicevic/576-NLP-Final-Project
./run_baseline.sh
```

**That's it!** The script will:
- ✅ Auto-activate the virtual environment
- ✅ Run the oracle baseline (500 questions)
- ✅ Save results to `baseline_results/`
- ⏱️ Take ~30-60 minutes
- 💰 Cost ~$0.50-$1.00

## What's Installed

All these packages are ready to use in the virtual environment:

```
✅ openai==1.35.1              - OpenAI API
✅ transformers==4.57.1         - Hugging Face
✅ torch==2.9.1                 - PyTorch
✅ sentence-transformers==5.1.2 - Dense retrieval
✅ numpy, nltk, tqdm, backoff   - Utilities
```

## Activate Virtual Environment Manually

If you want to run Python scripts manually:

```bash
# Activate
source venv/bin/activate

# You'll see (venv) in your prompt
(venv) $

# Run Python
python analyze_temporal_metadata.py

# Deactivate when done
deactivate
```

## Common Commands

### Run Baseline (Auto-activates venv)
```bash
./run_baseline.sh
```

### Run with Custom Data
```bash
./run_baseline.sh data/longmemeval_s_cleaned.json
```

### Analyze Temporal Metadata
```bash
source venv/bin/activate
python analyze_temporal_metadata.py
```

### Test Small Subset (Cheaper)
```bash
# Create 10-question test set
source venv/bin/activate
python -c "
import json
data = json.load(open('benchmarks/LongMemEval/data/longmemeval_oracle.json'))
json.dump(data[:10], open('benchmarks/LongMemEval/data/test_small.json', 'w'), indent=2)
"

# Run on small set
./run_baseline.sh data/test_small.json
```

## After Baseline Completes

### Evaluate Results
```bash
source venv/bin/activate
cd benchmarks/LongMemEval/src/evaluation

# Replace <file> with actual output file
python evaluate_qa.py gpt-4o-mini <hypothesis_file> ../../data/longmemeval_oracle.json
```

### View Metrics
Results will show:
- Overall accuracy
- Temporal-reasoning accuracy (most relevant!)
- Knowledge-update accuracy
- Multi-session, single-session breakdowns

## Directory Structure

```
576-NLP-Final-Project/
├── venv/                         ✅ Virtual environment
├── requirements.txt              ✅ Dependencies list
├── activate_venv.sh              ✅ Quick activation
├── run_baseline.sh               ✅ Baseline runner (auto-activates venv)
├── analyze_temporal_metadata.py  ✅ Analysis script
├── benchmarks/LongMemEval/       ✅ Benchmark code + data
├── baseline_results/             📁 Results go here
└── QUICK_START.md               📖 This file
```

## Troubleshooting

**"Command not found"**
```bash
chmod +x run_baseline.sh
```

**"No module named 'openai'"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Want to recreate venv**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## You're Ready!

Just run:
```bash
export OPENAI_API_KEY='your-key'
./run_baseline.sh
```

For detailed info, see:
- **VENV_SETUP.md** - Virtual environment details
- **BASELINE_GUIDE.md** - Complete baseline guide
- **SUMMARY.md** - Project overview
