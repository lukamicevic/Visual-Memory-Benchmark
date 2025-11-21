# START HERE 👋

## You're All Set to Run LongMemEval Baselines!

Everything has been set up for you. Follow these 3 simple steps:

---

## ⚡ Quick 3-Step Start

### Step 1: Get OpenAI API Key
1. Go to: **https://platform.openai.com/api-keys**
2. Sign up/log in
3. Click "Create new secret key"
4. Copy it (starts with `sk-proj-...`)

### Step 2: Set Your API Key
```bash
export OPENAI_API_KEY='sk-proj-your-actual-key-here'
```

### Step 3: Run Baseline
```bash
cd /Users/lukamicevic/576-NLP-Final-Project
./run_baseline.sh
```

**That's it!** 🎉

---

## ✅ What's Already Done

- ✅ **Virtual environment** created with all packages
- ✅ **LongMemEval data** downloaded (500 questions)
- ✅ **All dependencies** installed (openai, transformers, torch, etc.)
- ✅ **Scripts ready** to run experiments
- ✅ **Documentation** written

---

## 📊 What Will Happen

When you run `./run_baseline.sh`:

1. Virtual environment activates automatically
2. Connects to OpenAI API
3. Processes 500 questions with temporal metadata
4. Generates answers using GPT-4o-mini
5. Saves results to `baseline_results/`

**Time:** ~30-60 minutes  
**Cost:** ~$0.50-$1.00

---

## 📖 Documentation Quick Reference

- **START_HERE.md** ← You are here!
- **[QUICK_START.md](QUICK_START.md)** - Detailed quick start
- **[VENV_SETUP.md](VENV_SETUP.md)** - Virtual environment info
- **[BASELINE_GUIDE.md](BASELINE_GUIDE.md)** - Complete baseline guide
- **[SUMMARY.md](SUMMARY.md)** - Full project summary

---

## 🔧 Common Tasks

### Activate Virtual Environment Manually
```bash
source venv/bin/activate
```

### Analyze Temporal Metadata (Before Running)
```bash
source venv/bin/activate
python analyze_temporal_metadata.py
```

### Test with 10 Questions First (Cheaper)
```bash
source venv/bin/activate
python -c "
import json
d = json.load(open('benchmarks/LongMemEval/data/longmemeval_oracle.json'))
json.dump(d[:10], open('benchmarks/LongMemEval/data/test_10q.json', 'w'), indent=2)
"
./run_baseline.sh data/test_10q.json
```

### After Baseline Completes - Evaluate
```bash
source venv/bin/activate
cd benchmarks/LongMemEval/src/evaluation
python evaluate_qa.py gpt-4o-mini <hypothesis_file> ../../data/longmemeval_oracle.json
```

---

## 💡 Your Research Goal

**Current Baseline (what you're running now):**
- Temporal info as METADATA: `Session Date: 2023/04/10`
- External labels, not learned by model

**Your Goal (after baseline):**
- Encode time INTO data: `<T-5days>` or `[5 days ago]`
- Model learns temporal relationships directly

**Compare:** Metadata vs Encoded performance!

---

## 🎯 Next Steps After Baseline

1. **Run baseline** → Get metadata performance numbers
2. **Analyze results** → Which question types are hardest?
3. **Design encoding** → Choose temporal encoding approach
4. **Transform data** → Create encoded versions
5. **Re-run experiments** → Compare encoded vs metadata
6. **Write paper** → Report findings! 📝

---

## 🆘 Need Help?

**Virtual environment not working?**
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**API key not recognized?**
```bash
# Check if set
echo $OPENAI_API_KEY

# Set it again
export OPENAI_API_KEY='your-key'
```

**Want to understand the code?**
- See [BASELINE_GUIDE.md](BASELINE_GUIDE.md)
- See [SUMMARY.md](SUMMARY.md)
- Read the files in `benchmarks/LongMemEval/src/`

---

## 🚀 Ready to Start!

Just run these 2 commands:

```bash
export OPENAI_API_KEY='your-key-here'
./run_baseline.sh
```

Good luck with your temporal encoding research! 🎓

---

**Questions?** Check the documentation files above or examine the code.
