import argparse
import os
from typing import Dict, List

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.gemini import Gemini
from models.claude import Claude
from models.olmo import Olmo
from models.mistral import Mistral
from models.llama2 import Llama2
from models.llama3 import Llama3
from models.med42 import Med42
from models.openbiollm import OpenBioLLM
from models.biomistral import BioMistral
from models.biomedgpt import BioMedGPT
from models.alpacare import AlpaCare

from tqdm import tqdm
import random
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
import torch, gc

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
REQ_TIME_GAP = 15

class Evaluator:
    BASE_PROMPT = '''
    Spin or misrepresentation of study findings can be used to influence, positively, the interpretation of statistically nonsignificant randomized controlled trials (RCTs), for example, by emphasizing the apparent benefit of a secondary outcome or findings from a subgroup of patients.
    Does the following clinical trial abstract contain spin (yes/no)?
    Answer only with 'yes' or 'no'. Do not provide any explanations.
    
    Abstract: {ABSTRACT}
    '''
    MODELS_WITH_RATE_LIMIT = ["gemini_1.5_flash", "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku"]
    def __init__(self, model_name: str, output_path: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.output_path = output_path
        self.is_debug = is_debug

        self.dataset = None
        self.model = None
        self.max_new_tokens = self.__get_max_new_tokens()

        self.__load_dataset()
        self.__load_model()

    def __load_dataset(self) -> None:
        """
        This method loads the dataset (test split)

        :return dataset as a list of dictionaries
        """
        print("Loading the dataset...")
        eval_data_file_path = os.path.join(DATA_FOLDER_PATH, "spin_and_no_spin_abstracts.csv")
        dataset = load_csv_file(eval_data_file_path)

        # shuffle the dataset since it is ordered by pmcid
        random.seed(SEED) # set seed for reproducibility
        random.shuffle(dataset)

        # if test, only get 3 random examples
        if self.is_debug:
            # select random 3 instances
            dataset = random.sample(dataset, 3)
        self.dataset = dataset

    def __load_model(self) -> None:
        """
        This method loads the model requested for the task based on the model size.

        :return Model object
        """
        print("Loading the model...")
        model_class_mapping = {
            "gpt35": GPT35,
            "gpt4o": GPT4,
            "gpt4o-mini": GPT4,
            "gemini_1.5_flash": Gemini,
            "gemini_1.5_flash-8B": Gemini,
            "claude_3.5-sonnet": Claude,
            "claude_3.5-haiku": Claude,
            "olmo2_instruct-7B": Olmo,
            "olmo2_instruct-13B": Olmo,
            "mistral_instruct7B": Mistral,
            "llama2_chat-7B": Llama2,
            "llama2_chat-13B": Llama2,
            "llama2_chat-70B": Llama2,
            "llama3_instruct-8B": Llama3,
            "llama3_instruct-70B": Llama3,
            "med42-8B": Med42,
            "med42-70B": Med42,
            "openbiollm-8B": OpenBioLLM,
            "openbiollm-70B": OpenBioLLM,
            "biomistral7B": BioMistral,
            "biomedgpt7B": BioMedGPT,
            "alpacare-7B": AlpaCare
        }
        model_class = model_class_mapping[self.model_name]
        if "-" in self.model_name:
            type = model_name.split("-")[-1]
            self.model = model_class(model_type=type)
        else:
            self.model = model_class()

    def __get_max_new_tokens(self) -> int:
        """
        This method returns the maximum number of new tokens to add by the model

        :return maximum number of new tokens
        """
        return 50
    
    def __clean_text(self, text: str) -> str:
        """
        This method cleans the text by removing any leading/trailing whitespaces.
        Removes any punctuations and makes the text lowercase.

        :param text: input text to clean
        :return cleaned text
        """
        cleaned_text = text.strip().lower().translate(str.maketrans('', '', string.punctuation))
        cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ")
        # parse out the first instance of either 'yes' or 'no' from the text
        for word in cleaned_text.split():
            if word in ["yes", "no"]:
                return word
        return ""
    
    def __calculate_metrics(self, dataset: List[Dict]) -> Dict:
        """
        This method calculates accuracy, precision, recall, f1 score for the llm outputs against ground truth.
        Binary classification task with equal distribution of spin and no spin abstracts.

        :param dataset: list of dictionaries containing the abstracts and the model outputs
        :return dictionary containing the metrics
        """
        print("Calculating metrics...")
        df = pd.DataFrame(dataset)

        # check if any of results are errors
        if df["model_answer"].str.contains("Error").any():
            print("Some of the model outputs are errors. Cannot calculate the metrics.")
            return {}

        # check if column values have any Error or empty string values for model outputs
        if df["model_answer"].apply(lambda x: "Error" in x or x == "").any():
            print("Model's output has some 'Error' or empty string values. Removing these rows from the metrics...")
            # remove rows with 'Error' or empty string values
            df = df[df["model_answer"].apply(lambda x: "Error" not in x and x != "")]
            print(f"Number of rows after removing 'Error' or empty string values: {len(df)}")

        # calculate the metrics
        metrics = {}
        # convert the spin and model_answer to binary values
        df["ground_truth"] = df["abstract_type"].apply(lambda x: 1 if x == "spin" else 0)
        df["model_answer"] = df["model_answer"].apply(lambda x: 1 if x == "yes" else 0)

        # calculate the metrics (accuracy, precision, recall, f1 score)
        accuracy = accuracy_score(df["ground_truth"], df["model_answer"])
        precision = precision_score(df["ground_truth"], df["model_answer"])
        recall = recall_score(df["ground_truth"], df["model_answer"])
        f1 = f1_score(df["ground_truth"], df["model_answer"])

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        
        metrics["accuracy"] = accuracy
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

        return metrics

    def evaluate(self) -> None:
        """
        This method runs the evaluation on the dataset of abstracts with and without spin.
        We ask LLM to detect if the abstract has spin or not.
        The evaluation dataset is from Boutron et al., 2014.

        :return None
        """
        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running evaluation on the dataset")
        for _, example in enumerate(pbar):
            input = self.BASE_PROMPT.format(ABSTRACT=example["abstract"])
            output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
            
            example["model_raw_answer"] = output["response"] if "response" in output else "Error: No response from the model"
            example["model_answer"] = self.__clean_text(output["response"]) if "response" in output else "Error: No response from the model"
            example["model_log_probabilities"] = output["log_probabilities"] if "log_probabilities" in output else "Error: No response from the model"
            if self.model_name in self.MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting (free version/tier)
                time.sleep(REQ_TIME_GAP)
            results.append(example)

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name} to csv and json")

        # sort results by id
        results = sorted(results, key=lambda x: x["PMID"])

        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_detection_outputs.json"
        save_dataset_to_json(results, json_file_path)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_detection_outputs.csv"
        save_dataset_to_csv(results, csv_file_path)

        print(f"Model outputs saved to {json_file_path} and {csv_file_path}")

        # To calculate the classification metrics
        metrics = self.__calculate_metrics(results)

        # save the metrics to a file 
        metrics_file_path = f"{self.output_path}/{self.model_name}_detection_classification_metrics.json"
        save_dataset_to_json(metrics, metrics_file_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Evaluation of Detection of Spin in Clinical Trial Abstracts Using LLMs")

    parser.add_argument("--model", default="gpt4o", 
                        choices=["gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B"], 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--output_path", default="./eval_outputs", help="directory of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run random 3 instances from the dataset.")
    
    args = parser.parse_args()

    model_name = args.model
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for the Evaluator:")
    print(f"Model:        {model_name}")
    print(f"Output Path:  {output_path}")
    print(f"Is Debug:     {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    evaluator = Evaluator(model_name, output_path, is_debug)
    evaluator.evaluate()
    gc.collect()
    torch.cuda.empty_cache()