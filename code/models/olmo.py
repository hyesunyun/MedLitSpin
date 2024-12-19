from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

SEED = 42

class Olmo(Model):
    def __init__(self, model_type: str = "7B") -> None:
        super().__init__()
        set_seed(SEED)
        if model_type == "7B":
            # self.model_name = "allenai/OLMo-2-1124-7B-Instruct"
            self.model_name = "/projects/frink/models/olmo-2-1124-7b-instruct"
        else:
            # self.model_name = "allenai/OLMo-2-1124-13B-Instruct"
            self.model_name = "/projects/frink/models/olmo-2-1124-13b-instruct"
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 2048

    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
            inputs = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                result = self.model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, return_dict_in_generate=True, output_scores=True, return_legacy_cache=True)
            response = self.tokenizer.decode(result[0, inputs.shape[1]:], skip_special_tokens=True)
            transition_scores = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            )
            return {"response": response, "log_probabilities": transition_scores}
        except Exception as e:
            print("[ERROR]", e)