from .model import Model
from openai import OpenAI
from typing import Any, Dict
import os
from dotenv import load_dotenv
import time

REQ_TIME_GAP = 5 #
MAX_API_RETRY = 3

class GPT35(Model):
    def __init__(self) -> None:
        super().__init__()
        load_dotenv(override=True)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_context_length(self) -> int:
        return 16385
    
    def generate_output(self, input: str, max_new_tokens: int, temperature: int = 1, top_p: int = 1) -> Dict[str, Any]:
        """
        This method generates the output given the input

        :param input: input to the model
        :param max_new_tokens: maximum number of tokens to generate
        :param temperature: temperature parameter for the model

        :return output of the model
        """
        completion = None
        for _ in range(MAX_API_RETRY):
            try:
                completion = self.client.chat.completions.create(
                    model="gpt-3.5-turbo-0125", #16k tokens (https://platform.openai.com/docs/models#gpt-3-5-turbo)
                    messages=[
                        {
                            "role": "user", 
                            "content": input
                        }
                    ],
                    # TODO: currently set as default but should figure out temperature/top_p parameters
                    # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    logprobs=True,
                    top_logprobs=15
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
                print("[ERROR]", e)
                time.sleep(REQ_TIME_GAP)
                
        if completion is None:
            return {"error_message": "Error: GPT-3.5 API call failed."}
        else:
            return {"response": completion.choices[0].message.content, "log_probabilities": top_logprobs_dict}