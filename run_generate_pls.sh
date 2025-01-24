#!/bin/bash

# List of models to use

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
  "med42-70B"
  "biomistral7B"
  "openbiollm-8B"
  "openbiollm-70B"
  "llama2_chat-7B"
  "llama2_chat-13B"
  "llama2_chat-70B"
  "llama3_instruct-8B"
  "llama3_instruct-70B"
  "biomedgpt7B"
  "alpacare-7B"
)

# DEFAULT PROMPT TEMPLATE with 300 max new tokens

# # Loop through each model
# for model in "${models[@]}"; do
#   # Run the script with the current model
#   python3 code/generate_plain_language_summaries.py \
#     --model "$model" \
#     --output_path "code/pls_outputs/$model" \
#     --max_new_tokens 300 \
#     --prompt_template_name "default"
# done

# echo "####################################"

echo "Generating plain language summaries with ground truth spin/no spin labels..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/generate_plain_language_summaries_with_abstract_labels.py.py \
    --model "$model" \
    --label_mode "gold_label" \
    --output_path "code/pls_outputs/$model" \
    --max_new_tokens 300 \
    --prompt_template_name "default"
done

echo "####################################"

# Loop through each model
echo "Generating plain language summaries with model output's spin/no spin labels..."
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/generate_plain_language_summaries_with_abstract_labels.py.py \
    --model "$model" \
    --label_mode "model_output_label" \
    --output_path "code/pls_outputs/$model" \
    --max_new_tokens 300 \
    --prompt_template_name "default"
done