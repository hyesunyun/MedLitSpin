import argparse
import os
from typing import Dict, List

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.gemini import Gemini
from models.model import Model

from tqdm import tqdm
import random
import time

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
REQ_TIME_GAP = 20

# TODO: add personas to the prompts
class Generator:
    BASE_PROMPT = '''
    My fifth grader asked me what this passage means: {ABSTRACT}
    Help me summarize it for him, in plain language a fifth grader can understand.
    '''
    def __init__(self, model_name: str, output_path: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.output_path = output_path
        self.is_debug = is_debug

        self.dataset = None
        self.model = None
        self.max_new_tokens = self.__get_max_new_tokens()

        self.__load_dataset()
        self.__load_model()

    def __load_dataset(self) -> List[Dict]:
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
            # select random 3 pmids
            pmids = random.sample([example["PMID"] for example in dataset], 1)
            # get both spin and no spin abstracts for the selected pmids
            dataset = [example for example in dataset if example["PMID"] in pmids]
        self.dataset = dataset

    def __load_model(self) -> Model:
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
            "gemini_1.5_flash-8B": Gemini
        }
        model_class = model_class_mapping[self.model_name]
        if "-" in self.model_name:
            size = model_name.split("-")[-1]
            self.model = model_class(model_size=size)
        else:
            self.model = model_class()

    def __get_max_new_tokens(self) -> int:
        """
        This method returns the maximum number of new tokens to add by the model

        :return maximum number of new tokens
        """
        return 500 # arbitrary number that is big enough for the task

    def generate_pls(self) -> None:
        """
        This method generates the plain language summaries for the abstracts in the dataset using the specified model

        :return None
        """

        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running generation on the dataset")
        for _, example in enumerate(pbar):
            input = self.BASE_PROMPT.format(ABSTRACT=example["abstract"])
            output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
            
            example[f"plain_language_summary"] = output["response"] if "response" in output else "Error: No response from the model"
            if self.model_name == "gemini_1.5_flash" or self.model_name == "gemini_1.5_flash-8B":
                # add some default time gap to avoid rate limiting (free version)
                time.sleep(REQ_TIME_GAP)
            results.append(example)

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name} to csv and json")

        # sort results by id
        results = sorted(results, key=lambda x: x["PMID"])

        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_outputs.json"
        save_dataset_to_json(results, json_file_path)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_outputs.csv"
        save_dataset_to_csv(results, csv_file_path)

        print(f"Model outputs saved to {json_file_path} and {csv_file_path}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Generation of Plain Language Summaries from Abstracts Using LLMs")

    parser.add_argument("--model", default="gpt4o", choices=["gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", "gemini_1.5_flash-8B"], help="what model to run", required=True)
    parser.add_argument("--output_path", default="./pls_outputs", help="directory of where the outputs/results should be saved.")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run random 3 instances from the dataset.")
    
    args = parser.parse_args()

    model_name = args.model
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for the Generator:")
    print(f"Model:        {model_name}")
    print(f"Output Path:  {output_path}")
    print(f"Is Debug:     {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    generator = Generator(model_name, output_path, is_debug)
    generator.generate_pls()