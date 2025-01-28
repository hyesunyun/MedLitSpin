import argparse
import os

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
import time

from utils import load_csv_file, save_dataset_to_json, save_dataset_to_csv

DATA_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEED = 42
REQ_TIME_GAP = 15
DEFAULT_MAX_NEW_TOKENS = 300 # arbitrary number for default max tokens

# TODO: add details for different audience/length limits/etc.
class Generator:
    BASE_PROMPT = '''
    My fifth grader asked me what this passage means: {ABSTRACT}
    Help me summarize it for him, in plain language a fifth grader can understand.
    '''
    MODELS_WITH_RATE_LIMIT = ["gemini_1.5_flash", "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku"]
    def __init__(self, model_name: str, output_path: str, max_new_tokens: int, prompt_template_name: str, is_debug: bool = False) -> None:
        self.model_name = model_name
        self.output_path = output_path
        self.prompt_template_name = prompt_template_name
        self.is_debug = is_debug

        self.dataset = None
        self.model = None
        self.prompt_template = None
        self.max_new_tokens = max_new_tokens

        self.__load_prompt_template()
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
            # select random 3 pmids
            pmids = random.sample([example["PMID"] for example in dataset], 1)
            # get both spin and no spin abstracts for the selected pmids
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
    
    def __load_prompt_template(self) -> None:
        """
        This method loads the prompt template for the task

        :return None
        """
        print("Loading the prompt template...")
        template_mapping = {
            "default": self.BASE_PROMPT
        }
        try:
            self.prompt_template = template_mapping[self.prompt_template_name]
        except KeyError:
            raise ValueError("Template not found")

    def generate_pls(self) -> None:
        """
        This method generates the plain language summaries for the abstracts in the dataset using the specified model

        :return None
        """

        # run the task using specified model
        results = []
        pbar = tqdm(self.dataset, desc="Running generation on the dataset")
        for _, example in enumerate(pbar):
            input = self.prompt_template.format(ABSTRACT=example["abstract"])
            output = self.model.generate_output(input, max_new_tokens=self.max_new_tokens)
            
            example[f"plain_language_summary"] = output["response"] if "response" in output else "Error: No response from the model"
            if self.model_name in self.MODELS_WITH_RATE_LIMIT:
                # add some default time gap to avoid rate limiting (free version/tier)
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

    parser.add_argument("--model", default="gpt4o", 
                        choices=["gpt35", "gpt4o", "gpt4o-mini", "gemini_1.5_flash", 
                                 "gemini_1.5_flash-8B", "claude_3.5-sonnet", "claude_3.5-haiku", 
                                 "olmo2_instruct-7B", "olmo2_instruct-13B", "mistral_instruct7B", "llama2_chat-7B",
                                 "llama2_chat-13B", "llama2_chat-70B", "llama3_instruct-8B", "llama3_instruct-70B",
                                 "med42-8B", "med42-70B", "openbiollm-8B", "openbiollm-70B", "biomistral7B", "biomedgpt7B",
                                 "alpacare-7B"], 
                        help="what model to run", 
                        required=True)
    parser.add_argument("--output_path", default="./pls_outputs", help="directory of where the outputs/results should be saved.")
    parser.add_argument("--max_new_tokens", default=DEFAULT_MAX_NEW_TOKENS, type=int, help="maximum number of tokens to generate for the plain language summary")
    parser.add_argument("--prompt_template_name", default="default", help="name of the template to use for the prompt")
    # do --no-debug for explicit False
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, help="used for debugging purposes. This option will only run random 3 instances from the dataset.")
    
    args = parser.parse_args()

    model_name = args.model
    output_path = args.output_path
    max_new_tokens = args.max_new_tokens
    prompt_template_name = args.prompt_template_name
    is_debug = args.debug

    print("Arguments Provided for the Generator:")
    print(f"Model:        {model_name}")
    print(f"Output Path:  {output_path}")
    print(f"Max Output Tokens:   {max_new_tokens}")
    print(f"Prompt Template:     {prompt_template_name}")
    print(f"Is Debug:     {is_debug}")
    print()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Output path did not exist. Directory was created.")
    
    generator = Generator(model_name, output_path, max_new_tokens, prompt_template_name, is_debug)
    generator.generate_pls()