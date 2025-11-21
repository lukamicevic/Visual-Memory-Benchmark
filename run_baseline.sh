#!/bin/bash

# LongMemEval Baseline Runner
# This script runs the baseline evaluation with temporal information as METADATA
# Before temporal encoding experiments, this establishes performance with current approach

set -e

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✅ Virtual environment activated"
fi

echo "============================================"
echo "LongMemEval Baseline Experiment"
echo "Temporal Information as METADATA"
echo "============================================"
echo ""

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Configuration
DATA_FILE="${1:-data/longmemeval_oracle.json}"
MODEL="${2:-gpt-4o-mini}"
RETRIEVER="${3:-full-history-session}"
TOPK="${4:-100}"
HISTORY_FORMAT="${5:-json}"
USERONLY="${6:-false}"
READING_METHOD="${7:-con}"

echo "Configuration:"
echo "  Data: $DATA_FILE"
echo "  Model: $MODEL"
echo "  Retriever: $RETRIEVER"
echo "  Top-K: $TOPK"
echo "  Format: $HISTORY_FORMAT"
echo "  User Only: $USERONLY"
echo "  Reading: $READING_METHOD"
echo ""

# Navigate to generation directory
cd benchmarks/LongMemEval/src/generation

# Update the script with your API key
export OPENAI_KEY="$OPENAI_API_KEY"
export OPENAI_ORGANIZATION="${OPENAI_ORGANIZATION:-}"

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="../../../../baseline_results/${MODEL}_${READING_METHOD}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "Running baseline generation..."
echo "Output will be saved to: $OUT_DIR"
echo ""

# Run generation
python run_generation.py \
    --in_file "../../${DATA_FILE}" \
    --out_dir "$OUT_DIR" \
    --model_name "gpt-4o-mini-2024-07-18" \
    --model_alias "$MODEL" \
    --retriever_type "orig-session" \
    --openai_key "$OPENAI_KEY" \
    --topk_context "$TOPK" \
    --history_format "$HISTORY_FORMAT" \
    --useronly "$USERONLY" \
    --cot true

echo ""
echo "============================================"
echo "Generation complete!"
echo "Results saved to: $OUT_DIR"
echo ""
echo "To evaluate, run:"
echo "  cd benchmarks/LongMemEval/src/evaluation"
echo "  python evaluate_qa.py gpt-4o-mini <hypothesis_file> ../../data/longmemeval_oracle.json"
echo "============================================"
