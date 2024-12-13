#!/bin/bash

# # List of models to use
# models=("gpt35" "gpt4o" "gpt4o-mini" "gemini_1.5_flash" "gemini_1.5_flash-8B")

# # Loop through each model
# for model in "${models[@]}"; do
#   # Run the script with the current model
#   python3 code/generate_plain_language_summaries.py \
#     --model "$model" \
#     --output_path "code/pls_outputs/$model"
# done

  python3 code/generate_plain_language_summaries.py \
    --model gpt4o \
    --output_path "code/pls_outputs/gpt4o" \