import argparse
import os
from typing import Dict, List

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.gemini import Gemini
from models.model import Model

from tqdm import tqdm
import random
import pandas as pd
from statistics import mean
import time

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
REQ_TIME_GAP = 10

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

        # shuffle the dataset since it is ordered by pmcid
        random.seed(SEED) # set seed for reproducibility
        random.shuffle(dataset)

        # if test, only get 3 random examples
        if self.is_debug:
            # select random 3 pmids
            pmids = random.sample([example["PMID"] for example in dataset], 3)
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
        return 2
    
    def __calculate_differences(self, dataset: List[Dict]) -> Dict:
        """
        This method calculates the differences between the spin and no spin abstracts for each question
        and then calculates the average differences for each question and also the average across all questions.

        :param dataset: list of dictionaries containing the abstracts and the model outputs
        :return dictionary containing the differences
        """
        print("Calculating differences in scores between spin and no spin abstracts...")
        df = pd.DataFrame(dataset)
        # calculate the differences
        # get unique PMID values in a list
        pmids = df['PMID'].unique()

        # column names for the 5 questions
        column_names = ["benefit_answer", "rigor_answer", "importance_answer", "full_text_answer", "another_trial_answer"]

        diff_metrics = {}
        for col in column_names:
            column_dffs = []
            for pmid in pmids:
                # Get the rows for the current PMID
                pmid_rows = df[df['PMID'] == pmid]
                # get the 'spin' answer and the 'no_spin' answer
                spin_answer = int(pmid_rows.loc[pmid_rows['abstract_type'] == 'spin', col].values[0].strip())
                no_spin_answer = int(pmid_rows.loc[pmid_rows['abstract_type'] == 'no_spin', col].values[0].strip())
                # subtract the 'spin' answer from the 'no spin' answer
                diff = no_spin_answer - spin_answer
                
                column_dffs.append(diff)

            # Average all the differences for each column
            column_avg = mean(column_dffs)

            diff_metrics[f"{col}_avg"] = column_avg
            print(f"Average differences for '{col}': {column_avg}")

        # Average across all columns
        overall_avg = mean(diff_metrics.values())
        diff_metrics['overall_avg'] = overall_avg

        print(f"\nOverall average difference across all answers: {overall_avg}")

        return diff_metrics

    def evaluate(self) -> None:
        """
        This method runs the evaluation on the dataset of abstracts with and without spin.
        We ask LLMs questions about the abstracts and compare the responses.
        The evaluation dataset is from Boutron et al., 2014.

        :return None
        """

        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running evaluation on the dataset")
        for _, example in enumerate(pbar):
            for key in self.QUESTIONS:
                input = self.BASE_PROMPT.format(ABSTRACT=example["abstract"], QUESTION=self.QUESTIONS[key])
                output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
                
                example[f"{key}_answer"] = output["response"] if "response" in output else "Error: No response from the model"
                example[f"{key}_log_probabilities"] = output["log_probabilities"] if "log_probabilities" in output else "Error: No response from the model"
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

        # To calculate the difference between spin and no spin abstracts, we can calculate the difference in the scores for each question
        # and then calculate the average difference for each question and also the average across all questions
        diff_metrics = self.__calculate_differences(results)

        # save the differences to a file 
        diff_file_path = f"{self.output_path}/{self.model_name}_differences_metrics.json"
        save_dataset_to_json(diff_metrics, diff_file_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Evaluation of Interpreting Clinical Trial Results from Abstracts Using LLMs")

    parser.add_argument("--model", default="gpt4o", choices=["gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", "gemini_1.5_flash-8B"], help="what model to run", required=True)
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