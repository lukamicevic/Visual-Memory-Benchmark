#!/bin/bash

# Activation helper script for LongMemEval virtual environment
# Usage: source activate_venv.sh

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "✅ Virtual environment activated!"
echo ""
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo ""
echo "To deactivate, run: deactivate"
echo ""
