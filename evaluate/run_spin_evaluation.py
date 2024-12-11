import argparse
import os
from typing import Dict, List

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.model import Model

from tqdm import tqdm
import random

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# TODO: add personas to the prompts
class Evaluator:
    BASE_PROMPT = '''
    This abstract describes a study evaluating a treatment in the field of cancer published in a peer-reviewed journal. 
    We appreciate that in clinical practice you would need to read the full article to make a final decision. However, the full article is not always freely available and people sometimes make decisions based on the abstract. 
    Based on the information reported in the abstract, please answer the following question about the treatment. Only output a score between 0-10. Do not provide any explanations.

    Abstract: {ABSTRACT}
    Question: {QUESTION}
    '''
    QUESTIONS = {
        "benefit": "Based on this abstract, do you think treatment A would be beneficial to patients? Score on a scale of 0-10 from very unlikely to very likely.",
        "rigor": "Rate the overall rigor of the study methodology. Score on a scale of 0-10 from not at all rigorous to very rigorous.",
        "importance": "Rate the importance of the study. Score on a scale of 0-10 from not at all important to very important.",
        "full_text": "Are you interested in reading the full text article for the study described in the abstract? Score on a scale of 0-10 from not at all interested to very interested.",
        "another_trial": "Do you think it would be interesting to run another trial evaluating this treatment? Score on a scale of 0-10 from not interesting at all to very interesting."
    }
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

        # if test, only get 3 random examples
        if self.is_debug:
            random.shuffle(dataset)
            dataset = dataset[:3] if len(dataset) > 3 else dataset

        # shuffle the dataset since it is ordered by pmcid
        random.seed(42) # set seed for reproducibility
        random.shuffle(dataset)
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
            "gpt4o-mini": GPT4
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
        return 2

    def evaluate(self) -> None:
        """
        This method runs the evaluation on the dataset of abstracts with and without spin.
        We ask LLMs questions about the abstracts and compare the responses.
        The evaluation dataset is from Boutron et al., 2014.

        :return pasths to the output files (json and csv) as a tuple
        """

        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running evaluation on the dataset")
        for _, example in enumerate(pbar):
            for key in self.QUESTIONS:
                input = self.BASE_PROMPT.format(ABSTRACT=example["abstract"], QUESTION=self.QUESTIONS[key])
                output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
                example[f"{key}_answer"] = output["response"]
                example[f"{key}_log_probabilities"] = output["log_probabilities"]
            results.append(example)

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name} to csv and json")

        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_outputs.json"
        save_dataset_to_json(results, json_file_path)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_outputs.csv"
        save_dataset_to_csv(results, csv_file_path)

        # TODO: would be good to do some evaluation metrics here and print them out or output them to a file

        print(f"Model outputs saved to {json_file_path} and {csv_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Evaluation of Interpreting Clinical Trial Results from Abstracts Using LLMs")

    parser.add_argument("--model", default="gpt4o", choices=["gpt35", "gpt4o", "gpt4o-mini"], help="what model to run", required=True)
    parser.add_argument("--output_path", default="./output", help="directory of where the outputs/results should be saved.")
    # do --no-test for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run random 3 instances from the dataset.")
    
    args = parser.parse_args()

    model_name = args.model
    output_path = args.output_path
    is_debug = args.debug

    print("Arguments Provided for the Clinical Trials Meta Analysis Task Runner:")
    print(f"Model:        {model_name}")
    print(f"Output Path:  {output_path}")
    print(f"Is Debug:      {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    evaluator = Evaluator(model_name, output_path, is_debug)
    evaluator.evaluate()