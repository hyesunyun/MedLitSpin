# Generating unspun abstracts

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

The data is available in the data folder as csv file under the name "spin_abstracts.csv".

## :computer: Unspun Generation

Claude 4.0 sonnet was used to generate the unspun version of abstracts from abstracts with spin. 

`code/run_spin_unpun_generation.py`

Example script for running this task on claude 4.0 sonnet:
```bash
python3 code/run_spin_unpun_generation.py \
    --model claude_4.0-sonnet \
    --input_path data/spin_abstracts.csv
```

You can change the arguments to run different models and specify output paths.

> Arguments of `run_spin_unpun_generation.py`:
> - `--model`: model to evaluate ("gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku",  "claude_4.0-sonnet", "claude_4.0-opus")
> - `--input_path`: path to the input csv containing the abstracts with spin
> - `--output_path`: path of directory where json and csv files of the outputs from model and eval metrics should be saved.
> - `--debug`: adding this flag will only run 3 randomly sampled PubMed articles from dataset. This is for debugging purposes.

## Post Processing

Sometimes the unspun version has slightly different formatting as the original spin abstract like extra/missing extra lines, bolding of headers, or different/extra headers. Some of the unspun abstracts are edited slightly to keep the formatting uniform between the original spin and generated unspun version, making sure none of the content is changed.