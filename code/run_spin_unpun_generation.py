import argparse
import os
from typing import Dict, List

from models.gpt35 import GPT35
from models.gpt4 import GPT4
from models.gemini import Gemini
from models.claude35 import Claude35
from models.claude4 import Claude4


from tqdm import tqdm
import random
import pandas as pd
from statistics import mean
import time
import torch, gc
import re

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
REQ_TIME_GAP = 15

class Generator:
    BASE_PROMPT = '''
    This abstract of a randomized control trial contains spin.
    Spin in study findings can be used to influence, positively, the interpretation of statistically nonsignificant randomized controlled trials (RCTs), for example, by emphasizing the apparent benefit of a secondary outcome or findings from a subgroup of patients.
    Rewrite this abstract without spin, keeping word count between {LOWER_BOUND} and {UPPER_BOUND} and keep the headers the same. Make minimal changes and do not change sentences that do not contain spin.
    Only output the new abstract.

    Guidelines for writing an abstract without spin:
    In the Context section:
    a. Delete all information that could distort the understanding of the aim of the trial.
    i. The aim is to evaluate the treatment effect on a secondary outcome.
    ii. The aim is to evaluate the treatment effect for a subgroup.
    iii. The aim is to evaluate overall improvement.
    In the Methods section:
    a. Clearly report the primary outcome.
    b. According to space constraints, report all secondary outcomes evaluated in the Methods section or report no secondary outcome evaluated in the Methods section to avoid specifically highlighting statistically significant secondary outcomes.
    c. Delete information that could distort the understanding of the aim (eg, within-group comparison, modified population analysis, subgroup analysis).
    In the Results section:
    a. Delete subgroup analyses that were not prespecified, based on the primary outcome, and interpreted in light of the totality of prespecified subgroup analyses undertaken.
    b. Delete within-group comparisons.
    c. Delete linguistic spin.
    d. Report the results for the primary outcome with numbers in both arms (if possible with some measure of variability) with no wording of judgment.
    e. Report results for all secondary outcomes, for no secondary outcome, or for the most clinically important secondary outcome.
    f. Report safety data including reason for withdrawals; report treatment discontinuation when applicable.
    In the Conclusions section:
    a. Delete the author conclusion, and only add the following standardized conclusion: “the treatment A was not more effective than comparator B in patients with….”
    b. Specify the primary outcome in the conclusion when some secondary outcomes were statistically significant: “the treatment A was not more effective on overall survival than the comparator B in patients with….”

    Abstract: {ABSTRACT}
    '''

    MODELS_WITH_RATE_LIMIT = {"gemini_1.5_flash", "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku"}
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
        eval_data_file_path = os.path.join(DATA_FOLDER_PATH, "Spin_abstracts_test_new.csv")
        dataset = load_csv_file(eval_data_file_path)

        # shuffle the dataset since it is ordered by pmcid
        random.seed(SEED) # set seed for reproducibility
        random.shuffle(dataset)

        # if test, only get 3 random examples
        if self.is_debug:
            # select random 3 pmids
            pmids = random.sample([example["PMID"] for example in dataset], 3)
            # get both spun and unspun abstracts for the selected pmids
            dataset = [example for example in dataset if example["PMID"] in pmids]
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
            "claude_3.5-sonnet": Claude35,
            "claude_3.5-haiku": Claude35,
            "claude_4.0-sonnet": Claude4,
            "claude_4.0-opus": Claude4,
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
        return 1000
    

    def generate(self) -> None:
        """
        This method runs the generation of unspun abstracts on the dataset of spun abstracts.
        We ask LLMs questions about the abstracts and compare the responses.
        The evaluation dataset is from Boutron et al., 2014.

        :return None
        """

        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running evaluation on the dataset")
        for _, example in enumerate(pbar):
            word_count = len(example["Abstract"].split())
            lower_bound = word_count - 20
            upper_bound = word_count + 20
            input = self.BASE_PROMPT.format(LOWER_BOUND=lower_bound, UPPER_BOUND=upper_bound, ABSTRACT=example["Abstract"])
            output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)

            example[f"Unspun Abstract"] = output["response"] if "response" in output else "Error: No response from the model"
            # example[f"{key}_answer"] = self.__clean_text(output["response"]) if "response" in output else "Error: No response from the model"
            # example[f"log_probabilities"] = output["log_probabilities"] if "log_probabilities" in output else "Error: No response from the model"
            if self.model_name in self.MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting (free version/tier)
                time.sleep(REQ_TIME_GAP)
            results.append(example)

        # saving outputs to file
        print(f"Saving outputs from model - {self.model_name} to csv and json")

        # sort results by id
        # results = sorted(results, key=lambda x: x["PMID"])

        # convert into json
        json_file_path = f"{self.output_path}/{self.model_name}_unpun_abstract.json"
        save_dataset_to_json(results, json_file_path)

        # convert into csv
        csv_file_path = f"{self.output_path}/{self.model_name}_unpun_abstract.csv"
        save_dataset_to_csv(results, csv_file_path)

        print(f"Model outputs saved to {json_file_path} and {csv_file_path}")

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running Evaluation of Interpreting Clinical Trial Results from Abstracts Using LLMs")

    parser.add_argument("--model", default="gpt4o", 
                        choices=["gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku",  "claude_4.0-sonnet"], 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--output_path", default="../data/unspun_abstracts", help="directory of where the outputs/results should be saved.")
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
    
    generator = Generator(model_name, output_path, is_debug)
    generator.generate()
    gc.collect()
    torch.cuda.empty_cache()