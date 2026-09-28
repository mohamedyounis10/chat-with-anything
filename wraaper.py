import requests


class KaggleMistralLLM:

    def __init__(self, api_url, max_new_tokens=512, temperature=0.3):
        self.api_url = api_url.rstrip("/") + "/generate"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def invoke(self, prompt):
        response = requests.post(
            self.api_url,
            json={
                "prompt": prompt,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
