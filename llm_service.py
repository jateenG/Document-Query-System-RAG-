# File: llm_service.py

import logging
import requests
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_type: str = "ollama", model_name: str = "llama2", base_url: str = "http://localhost:11434"):
        """
        [cite_start]Initialize LLM service. [cite: 18]
        Args:
            [cite_start]model_type: 'ollama' for local or 'openai' for API [cite: 19]
            [cite_start]model_name: Model identifier [cite: 19]
            [cite_start]base_url: Base URL for Ollama server [cite: 19]
        """
        self.model_type = model_type
        self.model_name = model_name
        self.base_url = base_url

    def generate_response(self, prompt: str, context: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """Generate response using LLM with context."""
        if self.model_type == "ollama":
            return self._ollama_generate(prompt, context, temperature, max_tokens)
        else:
            [cite_start]raise ValueError(f"Unknown model type: {self.model_type}") [cite: 21]

    def _ollama_generate(self, prompt: str, context: str, temperature: float, max_tokens: int) -> str:
        """Generate using Ollama (local LLM)."""
        full_prompt = f"""Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"""

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return "Error generating response"