#!/bin/bash

evaluators=(
  # "gpt4o"
  # "claude_3.5-sonnet"
  # "llama2_chat-70B"
  "llama3_instruct-8B"
  # "llama3_instruct-70B"
  # "med42-70B"
  # "openbiollm-70B"
)

models=(
  "gpt35"
  "gpt4o"
  # "gpt4o-mini"
  # "gemini_1.5_flash"
  # "gemini_1.5_flash-8B"
  # "claude_3.5-sonnet"
  # "claude_3.5-haiku"
  # "olmo2_instruct-7B"
  # "olmo2_instruct-13B"
  # "mistral_instruct7B"
  # "med42-8B"
  # "biomistral7B"
  # "openbiollm-8B"
  # "llama2_chat-7B"
  # "llama2_chat-13B"
  # "llama3_instruct-8B"
  # "llama2_chat-70B"
  # "llama3_instruct-70B"
  # "med42-70B"
  # "openbiollm-70B"
  # "biomedgpt7B"
  # "alpacare-7B"
)

echo "Running evaluation for interpreting trial results from vanilla PLS using the evaluator models..."
for evaluator in "${evaluators[@]}"; do
  # Run the script with the current evaluator
  for model in "${models[@]}"; do
    python3 code/run_spin_interpretation_from_pls_evaluation.py \
      --model "$evaluator" \
      --input_path "code/pls_outputs/$model/${model}_outputs.csv" \
      --output_path "code/pls_outputs/_interpretation_eval_results/$evaluator/$model" --debug
  done
done
