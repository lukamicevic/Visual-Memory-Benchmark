#!/bin/bash

# Temporal Encoder Training Script
# Quick start script for training and evaluating temporal encoder

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================"
echo "Temporal Encoder Training Pipeline"
echo "============================================"
echo ""

# Configuration
DATA_PATH="${1:-benchmarks/LongMemEval/data/longmemeval_oracle.json}"
EPOCHS="${2:-5}"
BATCH_SIZE="${3:-16}"
BASE_MODEL="${4:-Alibaba-NLP/gte-base-en-v1.5}"

echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Base Model: $BASE_MODEL"
echo ""

# Create directories
mkdir -p temporal_encoding/checkpoints
mkdir -p temporal_retrieval_results

# Step 1: Train temporal encoder
echo "============================================"
echo "Step 1: Training Temporal Encoder"
echo "============================================"
echo ""

python3 temporal_encoding/train_temporal_encoder.py \
    --data_path "$DATA_PATH" \
    --base_model "$BASE_MODEL" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 2e-5 \
    --time_encoding sinusoidal \
    --fusion_method concat \
    --temporal_weight 0.3 \
    --window_weight 0.2 \
    --save_dir temporal_encoding/checkpoints

echo ""
echo "✅ Training complete!"
echo ""

# Find latest checkpoint
LATEST_CHECKPOINT=$(ls -t temporal_encoding/checkpoints/checkpoint_epoch_*.pt 2>/dev/null | head -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "❌ Error: No checkpoint found!"
    exit 1
fi

echo "Latest checkpoint: $LATEST_CHECKPOINT"
echo ""

# Step 2: Run temporal retrieval
echo "============================================"
echo "Step 2: Running Temporal Retrieval"
echo "============================================"
echo ""

python3 temporal_encoding/integrate_with_longmemeval.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --data "$DATA_PATH" \
    --top_k 100 \
    --output temporal_retrieval_results/temporal_oracle_top100.json

echo ""
echo "✅ Temporal retrieval complete!"
echo ""

# Step 3: Instructions for generation (requires OpenAI API key)
echo "============================================"
echo "Step 3: Generate Answers (Manual Step)"
echo "============================================"
echo ""
echo "To generate answers with GPT, run:"
echo ""
echo "  export OPENAI_API_KEY='your-key-here'"
echo "  cd benchmarks/LongMemEval/src/generation"
echo "  python3 run_generation.py \\"
echo "    --in_file ../../../../temporal_retrieval_results/temporal_oracle_top100.json \\"
echo "    --out_dir ../../../../temporal_results \\"
echo "    --model_name gpt-4o-mini-2024-07-18 \\"
echo "    --model_alias gpt-4o-mini \\"
echo "    --retriever_type oracle-session \\"
echo "    --openai_key \"\$OPENAI_API_KEY\" \\"
echo "    --topk_context 100 \\"
echo "    --history_format json \\"
echo "    --useronly false \\"
echo "    --cot true"
echo ""
echo "Then evaluate with:"
echo "  cd ../evaluation"
echo "  python3 evaluate_qa.py gpt-4o-mini <hypothesis_file> ../../data/longmemeval_oracle.json"
echo ""
echo "============================================"
echo "Training Pipeline Complete!"
echo "============================================"
