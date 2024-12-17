#!/bin/bash

# models=("gpt35" "gpt4o" "gpt4o-mini" "gemini_1.5_flash" "gemini_1.5_flash-8B" "claude_3.5-sonnet" "claude_3.5-haiku")
models=("olmo2-7B" "olmo2-13B" "mistral_instruct_7B" "llama2_chat-13B" "llama2_chat-70B" "llama3_instruct-8B" "llama3_instruct-70B")

# Loop through each model for two different evaluation tasks: spin detection and spin interpretation
echo "Running evaluation for detecting spin in abstracts of medical literature..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_detection_evaluation.py \
    --model "$model" \
    --output_path "code/eval_outputs/$model"
done

echo "Running evaluation for interpreting trial results..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_interpretation_evaluation.py \
    --model "$model" \
    --output_path "code/eval_outputs/$model"
done