#!/bin/bash

# List of models to evaluate
# models=("gpt35" "gpt4o" "gpt4o-mini" "gemini_1.5_flash" "gemini_1.5_flash-8B")
models=("gemini_1.5_flash" "gemini_1.5_flash-8B")

# Loop through each model
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_evaluation.py \
    --model "$model" \
    --output_path "code/eval_outputs/$model"
done