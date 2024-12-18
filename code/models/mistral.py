from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

SEED = 42

class Mistral(Model):
    def __init__(self) -> None:
        super().__init__()
        set_seed(SEED)
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 8192
        
    def __load_model(self): # context window size: 32k tokens but 8k tokens is recommended for best performance
        # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        model_name = "/projects/frink/models/mistral-7B-instruct-v0.1"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype="auto"
        )
        return model

    def __load_tokenizer(self):
        # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        model_name = "/projects/frink/models/mistral-7B-instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int, temperature: int = 0) -> Dict[str, Any]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            message = [
                {"role": "user", "content": input},
            ]
            model_inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, temperature=temperature, return_dict_in_generate=True, output_scores=True)
            response = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            transition_scores = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            )
            return {"response": response, "log_probabilities": transition_scores}
        except Exception as e:
            print("[ERROR]", e)
