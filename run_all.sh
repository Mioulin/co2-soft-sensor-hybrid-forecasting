#!/bin/bash
# Full pipeline runner
set -e
cd "$(dirname "$0")"
echo "=== CO2 Soft-Sensor Pipeline ==="
pip install -r requirements.txt -q
python scripts/run_pipeline.py
echo ""
echo "=== Tests ==="
pytest tests/ -v
echo ""
echo "=== Done ==="
echo "Figures: $(ls outputs/figures/*.png | wc -l) plots"
echo "Results: outputs/metrics/all_results.csv"
