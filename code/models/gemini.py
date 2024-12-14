import logging
from .model import Model
import google.generativeai as genai
from typing import Any, Dict
import os
from dotenv import load_dotenv

SEED = 42

class Gemini(Model):
    def __init__(self, model_type: str = "regular") -> None:
        super().__init__()
        if model_type == "8B":
            model_name = "gemini-1.5-flash-8b"
        else:
            model_name = "gemini-1.5-flash"
        load_dotenv(override=True)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model_name)

    def get_context_length(self) -> int:
        return 1000000
    
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
            completion = self.client.generate_content(
                input,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_new_tokens,
                    candidate_count=1
                    # seed=SEED, # seed is not supported in google-generativeai
                    # logprobs is not supported for the models we are using :(
                    # response_logprobs=True,
                    # logprobs=5,
                )
            )
        except Exception as e:
            logging.error(e)

        if completion is None:
            return {"error_message": "Error: Gemini API call failed."}
        else:
            return {"response": completion.text, "log_probabilities": None}