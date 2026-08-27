"""
Thin client for a local Ollama server running an open-weight, <=16B model.

Model choice: qwen2.5:7b-instruct (Apache-2.0 license, 7B params). See
DECISIONS.md for why Qwen2.5-7B over alternatives.
"""
import json
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b-instruct"


class LLMCallError(Exception):
    pass


def call_llm(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
             temperature: float = 0.0, timeout: int = 120, retries: int = 1) -> str:
    """Calls the local Ollama chat endpoint, requesting JSON-formatted output.
    Returns the raw text content. Raises LLMCallError after exhausting retries
    -- callers must catch this and abstain rather than crash (see score.py)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }

    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except Exception as e:  # noqa: BLE001 -- deliberately broad, see module docstring
            last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise LLMCallError(f"Ollama call failed after {retries + 1} attempts: {last_err}")
