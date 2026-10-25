"""Text-generation LLM (OpenAI-compatible): Ollama locally, or Groq / Gemini / OpenRouter."""
import json
import os
import re
import time
from functools import lru_cache

from openai import OpenAI

from . import config, metrics

PROVIDERS = {
    "ollama": dict(base_url=config.OLLAMA_URL, key="ollama", model=config.OLLAMA_MODEL),
    "groq": dict(base_url="https://api.groq.com/openai/v1", key=os.getenv("GROQ_API_KEY"),
                 model=os.getenv("GROQ_MODEL")),
    "gemini": dict(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                   key=os.getenv("GEMINI_API_KEY"), model=os.getenv("GEMINI_MODEL")),
    "openrouter": dict(base_url="https://openrouter.ai/api/v1", key=os.getenv("OPENROUTER_API_KEY"),
                       model=os.getenv("OPENROUTER_LLM_MODEL")),
}


def provider_info():
    p = PROVIDERS[config.LLM_PROVIDER]
    return f"{config.LLM_PROVIDER}:{p['model']}"


@lru_cache(maxsize=1)
def _client():
    p = PROVIDERS[config.LLM_PROVIDER]
    if not p["model"] or not p["key"]:
        raise RuntimeError(f"Set the API key and model name for LLM provider '{config.LLM_PROVIDER}' in .env")
    return OpenAI(base_url=p["base_url"], api_key=p["key"], timeout=300), p["model"]


def chat(messages, temperature=0.1, max_tokens=900):
    client, model = _client()
    t0 = time.time()
    r = client.chat.completions.create(model=model, messages=messages,
                                       temperature=temperature, max_tokens=max_tokens)
    u = getattr(r, "usage", None)
    metrics.record_llm(time.time() - t0,
                       getattr(u, "prompt_tokens", 0) if u else 0,
                       getattr(u, "completion_tokens", 0) if u else 0)
    return r.choices[0].message.content or ""


def ask_json(system, user):
    raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def is_available():
    try:
        client, model = _client()
        client.models.list()
        return True, provider_info()
    except Exception as e:  # noqa: BLE001
        return False, f"{provider_info()} - {e}"
