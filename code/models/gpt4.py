from .model import Model
from openai import OpenAI
from typing import Any, Dict
import os
import logging
from dotenv import load_dotenv

SEED = 42

class GPT4(Model):
    def __init__(self, model_type: str = "regular") -> None:
        super().__init__()
        if model_type == "mini":
            self.model_name = "gpt-4o-mini"
        else:
            self.model_name = "gpt-4o-2024-08-06"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_context_length(self) -> int:
        return 128000
    
    def generate_output(self, input: str, max_new_tokens: int, temperature: int = 0) -> Dict[str, Any]:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model
        
        :return output of the model
        """
        completion = None
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name, # 128,000 tokens (https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
                messages=[
                    {"role": "user", "content": input}
                ],
                temperature=temperature,
                max_completion_tokens=max_new_tokens,
                logprobs=True,
                top_logprobs=20,
                seed=SEED
            )
            
            top_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
            top_logprobs_dict = []
            for probs in top_logprobs:
                top_logprobs_dict.append({
                    "token": probs.token,
                    "bytes": probs.bytes,
                    "logprob": probs.logprob
                })
        except Exception as e:
            logging.error(e)
                
        if completion is None:
            return {"error_message": "Error: GPT-4o API call failed."}
        else:
            return {"response": completion.choices[0].message.content, "log_probabilities": top_logprobs_dict}