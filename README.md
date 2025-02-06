# LLMs and Spin in Medical Literature

*Code and Data for the paper "Caught in the Web of Words: Do LLMs Fall for Medical Spin?"* :page_facing_up:

**How susceptible are LLMs to spin in medical articles?**

## :hammer_and_wrench: SETUP

Create conda environment from the environment.yml: `conda env create -f environment.yml`

Activate the conda environment: `conda activate MedLitSpin`

## :bookmark_tabs: DATA

This project uses the cancer-related medical abstract dataset compiled by [Boutron et al. (2014)](https://ascopubs.org/doi/10.1200/JCO.2014.56.7503). This dataset comprises 60 real and synthetic abstracts illustrating cases of results reported with and without spin. 

More specifically, the dataset includes 30 "base" abstracts, i.e., real articles describing RCTs in the field of cancer which (1) report statistically nonsignificant differences for all primary outcomes measured for the intervention being studied, and, (2) contain spin in the results and conclusion sections of the abstract (intimating more positive findings). The "neutral" abstracts were created by manually editing each of the original 30 abstracts to remove spin.

The data is available in the `data` folder as csv file.

## :speech_balloon: MODELS

We use 22 LLMs, including both open and closed (proprietary) models, for our analyses.
We also include both general and specialized (biomedical) LLMs, and models spanning a range of parameter counts. 

| **Generalist Closed** | **Generalist Open** | **Biomedical Open** |
|---|---|---|
| Anthropic's Claude 3.5 Haiku and Sonnet, Google's Gemini 1.5 Flash and Flash 8B, OpenAI's GPT 3.5 175B, GPT4o, GPT4o-mini | Llama2 7B, 13B, and 70B Chat; Llama3 8B and 70B Instruct; Mistral 7B Instruct v0.1; OLMo2 7B and 13B Instruct | Alpacare 7B, BioMedGPT 7B, BioMistral 7B, Med42-v2 8B and 70B, OpenBioLLM 8B and 70B |

The generalist closed models can be run on CPU as they are accessed via APIs.
The generalist open and biomedical open models require GPUs to run.

## :test_tube: EXPERIMENTS

### Spin Detection

`code/run_spin_detection_evaluation.py`

Example script for running this task on GPT-4o:
```bash
python3 code/run_spin_detection_evaluation.py \
    --model gpt4o \
    --output_path code/eval_outputs/gpt4o
```

You can change the arguments to run different models and specify output paths.

> Arguments of `run_spin_detection_evaluation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

Script to replicate the paper experiment with 22 LLMs:
```bash
scripts/run_detection_evaluation.sh
```

### Interpretation of trial results for spin and unspun abstracts

`code/run_spin_interpretation_evaluation.py`

Example script for running this task on GPT 3.5:
```bash
python3 code/run_spin_interpretation_evaluation.py \
    --model gpt35 \
    --output_path code/eval_outputs/gpt35
```

> Arguments of `run_spin_interpretation_evaluation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

Script to replicate the paper experiment with 22 LLMs:
```bash
scripts/run_interpretation_evaluation.sh
```

### Simplifying spun and unspun abstracts

#### Generation

Generating the simplified (plain language) abstracts can be done by running `code/generate_plain_language_summaries.py`

Example script for generating simplified abstracts with Llama2 Chat 7B:
```bash
python3 code/generate_plain_language_summaries.py \
    --model llama2_chat-7B \
    --output_path code/pls_outputs/llama2_chat-7B \
    --max_new_tokens 300 \
    --prompt_template_name default
```

> Arguments of `generate_plain_language_summaries.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--max_new_tokens`: maximum number of tokens to generate for the simplified abstract.
> - `--prompt_template_name`: name of the template to use for the prompt. defaults to "default".
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

Script to replicate the paper experiment with 22 LLMs:
```bash
scripts/run_generate_pls.sh
```

#### Evaluation

Running the interpretation task with simplified (plain language) abstracts can be done by running `code/run_spin_interpretation_from_pls_evaluation.py`

Example script with Claude 3.5 Sonnet as evaluator on Llama2 Chat 7B outputs:
```bash
python3 code/run_spin_interpretation_from_pls_evaluation.py \
      --model claude_3.5-sonnet \
      --input_path code/pls_outputs/llama2_chat-7B/llama2_chat-7B_outputs.csv \
      --output_path code/pls_outputs/_interpretation_eval_results/$evaluator/llama2_chat-7B
```

> Arguments of `run_spin_interpretation_from_pls_evaluation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--input_path`: directory of where the input data (csv) is stored.
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

Script to replicate the paper experiment with Claude 3.5 Sonnet & GPT4o Mini as evalutors evaluating simplified abstracts from all 22 LLMs:
```bash
scripts/run_pls_evaluation.sh
```

### Reducing the effect of spin

#### Adding reference or model output spin labels

The interpretation task with additional information about the given abstract can be done by running `code/run_spin_interpretation_evaluation_with_abstract_labels.py`

Example script for interpreting results with spin labels from model outputs:
```bash
python3 code/run_spin_interpretation_evaluation_with_abstract_labels.py \
    --model llama3_instruct-70B \
    --label_mode model_output_label \
    --output_path code/eval_outputs/llama3_instruct-70B
```

> Arguments of `run_spin_interpretation_evaluation_with_abstract_labels.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--label_mode`: which abstract labels will be used for prompting. options are "gold_label" and "model_output_label".
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

#### Joint prompting both detection and interpretation

The interpretation task with spin detection can be done by running `code/run_spin_combined_detection_interpretion_evaluation.py`

Example script for interpreting results after detecting spin in the same prompt:
```bash
python3 code/run_spin_combined_detection_interpretion_evaluation.py \
    --model med42-70B \
    --output_path code/eval_outputs/med42-70B
```
> Arguments of `run_spin_combined_detection_interpretion_evaluation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B")
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

Script to replicate the paper experiment running all mitigation strategies:
```bash
scripts/run_evaluation_mitigation_strategies.sh
```