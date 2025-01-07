#!/bin/bash

# List of models to use
models=(
  "gpt4o" 
  "gpt4o-mini" 
  "claude_3.5-sonnet"
  "llama3_instruct-8B"
  "llama2_chat-70B"
  "llama3_instruct-70B"
  "med42-70B"
  "openbiollm-70B"
)

# DEFAULT PROMPT TEMPLATE with 300 max new tokens

# Loop through each model
for model in "${models[@]}"; do
  # Run the script with the current model
  python3 code/generate_plain_language_summaries.py \
    --model "$model" \
    --output_path "code/pls_outputs/$model" \
    --max_new_tokens 300 \
    --prompt_template_name "default"
done