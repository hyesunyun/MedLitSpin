from .model import Model
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Dict, Any

SEED = 42

class AlpaCare(Model):
    def __init__(self, model_type: str = "7B") -> None:
        super().__init__()
        set_seed(SEED)
        if model_type == "7B":
            # self.model_name = "xz97/AlpaCare-llama2-7b"
            self.model_name = "/projects/frink/models/alpaca-7b"
        else:
            # self.model_name = "xz97/AlpaCare-llama2-13b"
            self.model_name = "/projects/frink/models/alpaca-13b"
        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

    def get_context_length(self) -> int:
        return 4000
        
    def __load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def generate_output(self, input: str, max_new_tokens: int) -> Dict[str, Any]:
        """
        This method generates the output given the input. Uses chat template for input.

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate

        :return output of the model
        """
        try:
            message = """
            <s>[INST] <<SYS>>
            {{ system_prompt }}
            <</SYS>>

            {{ user_message }} [/INST]
            """.format(system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.", 
                       user_message=input)
            model_inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                result = self.model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_scores=True)
            response = self.tokenizer.decode(result[0, model_inputs.shape[1]:], skip_special_tokens=True)
            transition_scores = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            )
            return {"response": response, "log_probabilities": transition_scores}
        except Exception as e:
            print("[ERROR]", e)