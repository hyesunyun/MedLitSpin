from abc import ABC, abstractmethod
from typing import Any, Dict

class Model(ABC):
    @abstractmethod
    def generate_output(self, input: str, max_new_tokens: int, temperature: int = 1, top_p: int = 1) -> Dict[str, Any]:
        """
        This method must be overridden

        :abstract

        """
        pass

    @abstractmethod 
    def get_context_length(self) -> int:
        """
        This method must be overridden

        :abstract

        """
        pass