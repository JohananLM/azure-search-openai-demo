from typing import Any


    

class Approach:
    def run(self, q: str, overrides: dict[str, Any]) -> Any:
        raise NotImplementedError

class ChatApproach(Approach):
    def run(self, q:str, overrides:dict[str,Any]) -> Any:
        raise NotImplementedError