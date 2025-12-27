#!/bin/bash
# Reproduce the Headline Result (N=30 Sweep)
# Usage: ./run_sweep.sh

echo "Starting Gravity Agents Final Sweep (Seeds 2000-2029)..."
python3 experiment_final_sweep.py
echo "Sweep Complete. Results saved to final_classified_results.json"
