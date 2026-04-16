# Semi-Synthetic Dataset

*Code and Data for generating unspun version of abstracts from RCT abstracts containing spin*

## :hammer_and_wrench: SETUP

Create conda environment from the environment.yml: `conda env create -f environment.yml`

Activate the conda environment: `conda activate MedLitSpin`

## :bookmark_tabs: DATA

This project uses medical abstract dataset from 3 different medical fields: [Orthopaedic](https://osf.io/89eky), [Emergency Medicine](https://osf.io/c8np7), and [Obesity](https://osf.io/gprzw). Each containing 36, 213, and 45 abstracts respectively, giving 295 abstracts in total. These datasets contains abstracts of Randomized Controlled Trials (RCTs) that illustrates cases of results reported with spin.

The datasets include abstracts of RCTs with 3 different types of spin: (1) spin due to selective reporting, (2) spin in abstract title/results, (3) spin in abstract conclusion.

### Data Processing

Only the content of the abstract is kept, all extra information like reference number, keywords, and level of evidence are all left out.

To align with the type of data used in *"Caught in the Web of Words: Do LLMs Fall for Medical Spin?"*, only RCTs with (2) spin in abstract title/results and (3) spin in abstract conclusion are used. We are left with 150 abstracts in total.

The data is available in the data folder as csv file under the name ["spin_abstracts.csv"](https://github.com/hyesunyun/MedLitSpin/blob/main/data/spin_abstracts.csv).

### :computer: Unspun Generation

GPT-4o was used to generate the unspun version of each abstracts with spin. 

`code/run_spin_unpun_generation.py`

Example script for running this task on GPT-4o:
```bash
python3 code/run_spin_unpun_generation.py \
    --model gpt4o \
    --input_path data/spin_abstracts.csv
```

You can change the arguments to run different models and specify output paths.

> Arguments of `run_spin_unpun_generation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku",  "claude_4.0-sonnet", "claude_4.0-opus")
> - `--input_path`: path to the input csv containing the abstracts with spin
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved. defaults to "../data/unspun_abstracts"
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

The output file from this generation process can be found in ["gpt4o_generated_no_spin_abstracts.csv"](https://github.com/hyesunyun/MedLitSpin/blob/main/data/unspun_abstracts/gpt4o_generated_no_spin_abstracts.csv)

### :floppy_disk: Post Processing

Sometimes the model-generated unspun versions had slightly different formatting from the original abstracts such as extra/missing lines, bolding of headers, or different/extra headers. Some of the unspun abstracts were manually edited slightly to keep the formatting uniform between the original spun and generated unspun versions, making sure none of the content is changed.

The formatted dataset can be found in ["gpt4o_generated_no_spin_abstracts_formatted.csv"](https://github.com/hyesunyun/MedLitSpin/blob/main/data/unspun_abstracts/gpt4o_generated_no_spin_abstracts_formatted.csv)

## :test_tube: EXPERIMENTS

We conducted the same experiments as the original dataset using this extended (semi-synthetic) dataset.

The outputs from spin detection, interpretation, and reducing the effect of spin analysis can be found in "data/unspun_abstracts/analysis"[https://github.com/hyesunyun/MedLitSpin/tree/main/data/unspun_abstracts/analysis] directory.

Outputs from running experiments with simplified or plain language abstracts can be found in ["code/pls_outputs/extended_dataset"](https://github.com/hyesunyun/MedLitSpin/tree/main/code/pls_outputs/extended_dataset).

All plots from experiments with the semi-synthetic dataset can be found ["code/plots/extended_dataset"](https://github.com/hyesunyun/MedLitSpin/tree/main/code/plots/extended_dataset)
