#!/bin/bash

# Closed-source models (API)
# models=(
#   "gpt35"
#   "gpt4o"
#   "gpt4o-mini"
#   "gemini_1.5_flash"
#   "gemini_1.5_flash-8B"
#   "claude_3.5-sonnet"
#   "claude_3.5-haiku"
# )

models=(
  # "olmo2_instruct-7B"
  # "olmo2_instruct-13B"
  # "mistral_instruct7B"
  # "med42-8B"
  # "biomistral7B"
  # "openbiollm-8B"
  "llama2_chat-7B"
  # "llama2_chat-13B"
  # "llama3_instruct-8B"
  ########################
  # "llama2_chat-70B"
  # "llama3_instruct-70B"
  # "med42-70B"
  # "openbiollm-70B",
  ########################
  # "alpacare-7B"
  # "alpacare-13B"
)

# models=(
#   "biomedgpt7B"
# )

# Loop through each model for two different evaluation tasks: spin detection and spin interpretation
echo "Running evaluation for detecting spin in abstracts of medical literature..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_detection_evaluation.py \
    --model "$model" \
    --output_path "code/eval_outputs/$model"
done

echo "####################################"

echo "Running evaluation for interpreting trial results..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_interpretation_evaluation.py \
    --model "$model" \
    --output_path "code/eval_outputs/$model"
done
