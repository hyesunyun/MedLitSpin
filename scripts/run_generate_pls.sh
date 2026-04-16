#!/bin/bash

# List of models to use

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

# DEFAULT PROMPT TEMPLATE with 300 max new tokens

echo "Original Dataset (30 pairs)"
# Loop through each model
echo "Generating plain language summaries..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 ../code/generate_plain_language_summaries.py \
    --model "$model" \
    --output_path "../code/pls_outputs/original_dataset/$model" \
    --max_new_tokens 300 \
    --prompt_template_name "default"
done

echo "########################################################################"

echo "Extended Dataset (150 pairs)"
echo "Generating plain language summaries for extended dataset..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 ../code/generate_plain_language_summaries.py \
    --model "$model" \
    --output_path "../code/pls_outputs/extended_dataset/$model" \
    --max_new_tokens 300 \
    --prompt_template_name "default" \
    --input_file ../data/unspun_abstracts/gpt4o_generated_no_spin_abstracts_formatted.csv
done