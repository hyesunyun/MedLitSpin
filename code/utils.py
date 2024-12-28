
import csv
import json
from typing import Dict, List, Optional
import numpy as np

def load_csv_file(file_path: str) -> List[Dict]:
    """
    This method loads a CSV file and returns the data as a list of dictionaries

    :param file_path: path to the CSV file

    :return data as a list of dictionaries
    """
    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data

def save_dataset_to_json(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset (dictionary) in json file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(dataset, file, indent=4)


def save_dataset_to_csv(dataset: List[Dict], file_path: str, columns_to_drop: Optional[List[str]] = None) -> None:
    """
    This method saves a dataset (dictionary) in csv file to the data folder

    :param dataset: dataset to save
    :param file_path: name of the dataset to save

    """
    if columns_to_drop is not None:
        dataset = [{k: v for k, v in d.items() if k not in columns_to_drop} for d in dataset]
    keys = dataset[0].keys()
    with open(file_path, "w", newline='', encoding='utf-8') as file:
        dict_writer = csv.DictWriter(file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)

def format_transition_scores(tokenizer, generated_tokens, transition_scores) -> List[Dict]:
    """
    This method formats the transition scores for the generated tokens

    :param tokenizer: tokenizer
    :param generated_tokens: generated tokens
    :param transition_scores: transition scores

    :return formatted transition scores
    """
    formatted_scores = []
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        tok_numpy = tok.numpy()
        score_numpy = score.numpy()
        formatted_scores.append({"token": tok_numpy.item(), "token_string": tokenizer.decode(tok), "logits": score_numpy.item(), "probability": np.exp(score_numpy).item()})
    return formatted_scores
