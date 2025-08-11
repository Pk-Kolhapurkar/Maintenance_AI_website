import os
from openai import OpenAI, AzureOpenAI
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

load_dotenv()

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

class LLM:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.model_name: str = model_name or os.getenv("MODEL_NAME") or config.get('model_name', "")
        inference_engine = config.get('inference_engine', 'openai').lower()

        if inference_engine == 'azure':
            azure_endpoint = api_url or os.getenv("AZURE_OPENAI_ENDPOINT") or ""
            self.client = AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY") or "",
                api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "",
                azure_endpoint=azure_endpoint
            )
        else:
            self.client = OpenAI(
                base_url=api_url or os.getenv("API_URL") or "",
                api_key=api_key or os.getenv("API_KEY") or ""
            )

    def query_model(self, prompt: str, system_message: Optional[str] = None) -> str:
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages  # type: ignore
        )
        return response.choices[0].message.content or ""

client = LLM(
    api_url=os.getenv(
        "AZURE_OPENAI_ENDPOINT" if config.get('inference_engine') == 'azure' else "API_URL"
    ),
    api_key=os.getenv(
        "AZURE_OPENAI_API_KEY" if config.get('inference_engine') == 'azure' else "API_KEY"
    ),
    model_name=os.getenv(
        "AZURE_OPENAI_MODEL_NAME" if config.get('inference_engine') == 'azure' else "MODEL_NAME"
    )
)

if __name__ == "__main__":
    user_prompt = "Hello, how are you?"
    response = client.query_model(user_prompt)
    print(response)
