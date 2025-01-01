from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any
from utils import format_transition_scores
import logging

SEED = 42

class BioMedGPT(Model):
    PROMPT = """<s>[INST] {instruction} [/INST]""" # llama2 chat template which this model is based on
    def __init__(self) -> None:
        super().__init__()
        set_seed(SEED)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 2048
        
    def __load_model(self):
        # model_name = "PharMolix/BioMedGPT-LM-7B"
        model_name = "/projects/frink/models/biomedgpt-7b"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        ) # float16 based on config.json
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

        # print model's dtype and device
        print(f"Model's dtype: {model.dtype}")
        print(f"Model's device: {model.device}")
        print(f"Model's device map: {model.hf_device_map}")
        print()

        return model

    def __load_tokenizer(self):
        # model_name = "PharMolix/BioMedGPT-LM-7B"
        model_name = "/projects/frink/models/biomedgpt-7b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int) -> Dict[str, Any]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            text_input = self.PROMPT.format(instruction=input)
            model_inputs = self.tokenizer(text_input, return_tensors="pt").input_ids.to(self.model.device)
            with torch.no_grad():
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True)
            response = self.tokenizer.decode(result.sequences[0, model_inputs.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            transition_scores = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            ).cpu()
            transition_scores = format_transition_scores(self.tokenizer, result.sequences[:, model_inputs.shape[1]:].cpu(), transition_scores)

            return {"response": response, "log_probabilities": transition_scores}
        except Exception as e:
            logging.error("[ERROR] %s", e)
            return {"error_message": f"Error: {e}"}