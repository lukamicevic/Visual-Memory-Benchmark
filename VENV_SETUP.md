# Virtual Environment Setup - COMPLETE ✅

## What Was Done

I've created a Python virtual environment with all required dependencies for LongMemEval experiments.

### Files Created

1. **`venv/`** - Python virtual environment
   - Isolated Python environment
   - All packages installed locally

2. **`requirements.txt`** - Dependency list
   - All required packages with versions
   - Easy to recreate environment

3. **`activate_venv.sh`** - Quick activation script
   - Helper to activate the venv
   - Shows Python version and location

### Installed Packages ✅

**Core Dependencies:**
- ✅ openai==1.35.1 - OpenAI API client
- ✅ transformers==4.57.1 - Hugging Face transformers
- ✅ torch==2.9.1 - PyTorch
- ✅ tiktoken==0.12.0 - OpenAI tokenizer
- ✅ numpy==1.26.3 - Numerical computing
- ✅ nltk==3.9.1 - Natural language toolkit
- ✅ tqdm==4.66.4 - Progress bars
- ✅ backoff==2.2.1 - Retry logic

**Retrieval Dependencies:**
- ✅ sentence-transformers==5.1.2 - Dense retrievers
- ✅ rank-bm25 - BM25 retrieval
- ✅ scikit-learn==1.7.2 - ML utilities

## How to Use

### Every Time You Work on This Project

**Option 1: Use the activation script**
```bash
cd /Users/lukamicevic/576-NLP-Final-Project
source activate_venv.sh
```

**Option 2: Activate manually**
```bash
cd /Users/lukamicevic/576-NLP-Final-Project
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when activated.

### Running Scripts with the Venv

The virtual environment is now **automatically activated** by `run_baseline.sh`, so you can just:

```bash
cd /Users/lukamicevic/576-NLP-Final-Project

# Set your API key
export OPENAI_API_KEY='your-key-here'

# Run baseline (venv activated automatically)
./run_baseline.sh
```

### Deactivating

When you're done working:
```bash
deactivate
```

## Verifying Installation

To check everything is working:

```bash
source venv/bin/activate
python -c "import openai, transformers, torch; print('✅ All packages working!')"
```

Should print: `✅ All packages working!`

## Troubleshooting

**If activation fails:**
```bash
# Recreate the venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**If packages are missing:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Check what's installed:**
```bash
source venv/bin/activate
pip list
```

## Directory Structure

```
576-NLP-Final-Project/
├── venv/                    # ✅ Virtual environment (don't commit to git)
│   ├── bin/activate        # Activation script
│   ├── lib/                # Installed packages
│   └── ...
├── requirements.txt         # ✅ Package dependencies
├── activate_venv.sh         # ✅ Quick activation helper
├── run_baseline.sh          # ✅ Auto-activates venv
└── ...
```

## Benefits of Virtual Environment

✅ **Isolated** - Packages don't conflict with system Python
✅ **Reproducible** - Same versions on any machine
✅ **Clean** - Easy to delete and recreate
✅ **Portable** - Share requirements.txt with others

## Next Steps

You're ready to run experiments! Just:

1. Get your OpenAI API key from https://platform.openai.com/api-keys
2. Set it: `export OPENAI_API_KEY='your-key'`
3. Run: `./run_baseline.sh`

The virtual environment will activate automatically! 🚀
