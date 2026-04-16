#!/bin/bash

models=(
  "alpacare-7B"
  "biomedgpt7B"
  "biomistral7B"
  "claude_3.5-haiku"
  "claude_3.5-sonnet"
  "gemini_1.5_flash"
  "gemini_1.5_flash-8B"
  "gpt35"
  "gpt4o"
  "gpt4o-mini"
  "llama2_chat-13B"
  "llama2_chat-70B"
  "llama2_chat-7B"
  "llama3_instruct-70B"
  "llama3_instruct-8B"
  "med42-70B"
  "med42-8B"
  "mistral_instruct7B"
  "olmo2_instruct-13B"
  "olmo2_instruct-7B"
  "openbiollm-70B"
  "openbiollm-8B"
)

echo "Original Dataset (30 pairs)"
echo "Running evaluation for interpreting trial results..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 ../code/run_spin_interpretation_evaluation.py \
    --model "$model" \
    --output_path "../code/eval_outputs/$model"
done

echo "########################################################################"

echo "Extended Dataset (150 pairs)"
echo "Running evaluation for interpreting trial results..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 ../code/run_spin_interpretation_evaluation.py \
    --model "$model" \
    --output_path "../data/unspun_abstracts/analysis/$model" \
    --input_file ../data/unspun_abstracts/gpt4o_generated_no_spin_abstracts_formatted.csv
done