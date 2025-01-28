#!/bin/bash

models=(
  "gpt35"
  "gpt4o"
  "gpt4o-mini"
  "gemini_1.5_flash"
  "gemini_1.5_flash-8B"
  "claude_3.5-sonnet"
  "claude_3.5-haiku"
  "olmo2_instruct-7B"
  "olmo2_instruct-13B"
  "mistral_instruct7B"
  "med42-8B"
  "biomistral7B"
  "openbiollm-8B"
  "llama2_chat-7B"
  "llama2_chat-13B"
  "llama3_instruct-8B"
  "llama2_chat-70B"
  "llama3_instruct-70B"
  "med42-70B"
  "openbiollm-70B"
  "biomedgpt7B"
  "alpacare-7B"
)

echo "Running evaluation for interpreting trial results from vanilla PLS..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_interpretation_from_pls_evaluation.py \
    --model "$model" \
    --input_path "code/pls_outputs/$model/${model}_outputs.csv" \
    --output_path "code/eval_outputs/$model"
done

echo "####################################"

echo "Running evaluation for interpreting trial results from PLS generated with gold spin labels..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_interpretation_from_pls_evaluation.py \
    --model "$model" \
    --input_path "code/pls_outputs/$model/${model}_gold_labelled_outputs.csv" \
    --output_path "code/eval_outputs/$model"
done

echo "####################################"

echo "Running evaluation for interpreting trial results from PLS generated with model output spin labels..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/run_spin_interpretation_from_pls_evaluation.py \
    --model "$model" \
    --input_path "code/pls_outputs/$model/${model}_model_output_labelled_outputs.csv" \
    --output_path "code/eval_outputs/$model"
done
