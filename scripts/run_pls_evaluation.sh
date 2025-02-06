#!/bin/bash

evaluators=(
  "claude_3.5-sonnet"
  "gpt4o-mini"
)

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

echo "Running evaluation for interpreting trial results from plan language summaries (PLS) using the evaluator models..."
for evaluator in "${evaluators[@]}"; do
  # Run the script with the current evaluator
  for model in "${models[@]}"; do
    python3 code/run_spin_interpretation_from_pls_evaluation.py \
      --model "$evaluator" \
      --input_path "code/pls_outputs/$model/${model}_outputs.csv" \
      --output_path "code/pls_outputs/_interpretation_eval_results/$evaluator/$model"
  done
done
