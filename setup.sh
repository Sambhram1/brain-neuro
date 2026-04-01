#!/usr/bin/env bash
set -e

echo "=== TRIBE v2 Brain Analyzer Setup ==="

# 1. Clone tribev2 if not already present
if [ ! -d "tribev2" ]; then
  git clone https://github.com/facebookresearch/tribev2
fi

# 2. Install tribev2 and its deps
pip install -e ./tribev2

# 3. Install app deps
pip install streamlit numpy

echo ""
echo "=== Done. Run with: ==="
echo "  streamlit run app.py"
