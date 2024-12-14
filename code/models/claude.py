from .model import Model
import anthropic
from typing import Any, Dict
import os
import logging
from dotenv import load_dotenv

class Claude(Model):
    def __init__(self, model_type: str = "sonnet") -> None:
        super().__init__()
        if model_type == "haiku":
            self.model_name = "claude-3-5-haiku-20241022"
        else:
            self.model_name = "claude-3-5-sonnet-20241022"
        logging.basicConfig(level=logging.ERROR)
        load_dotenv(override=True)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def get_context_length(self) -> int:
        return 200000
    
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
            completion = self.client.messages.create(
                model=self.model_name, # 200,000 tokens (https://docs.anthropic.com/en/docs/about-claude/models)
                max_tokens=max_new_tokens,
                messages=[
                    {"role": "user", "content": input}
                ],
                temperature=temperature
            )
        except Exception as e:
            logging.error(e)
                
        if completion is None:
            return {"error_message": "Error: Anthropic Claude API call failed."}
        else:
            return {"response": completion.content[0].text, "log_probabilities": None} # no log probs from Anthropic API