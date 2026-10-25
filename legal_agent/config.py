"""Central settings, read from .env (see .env.example)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
EMB_PATH = DATA_DIR / "emb.npy"
SAMPLES_DIR = DATA_DIR / "samples"
KNOWLEDGE_DIR = ROOT / "knowledge"          # the OKF bundle
REPORTS_DIR = ROOT / "reports"
UPLOADS_DIR = DATA_DIR / "uploads"

# ---- LLM (text generation) ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")          # ollama | groq | gemini | openrouter
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

# ---- Jev (TypeSafe System One decision model) ----
# typesafe   : official API (needs TYPESAFE_API_KEY)
# openrouter : Jev served by OpenRouter (needs OPENROUTER_API_KEY)
# vercel     : Jev served by Vercel AI Gateway (needs AI_GATEWAY_API_KEY)
# server     : an open Jev-compatible decision model on your machine (ollajev), at JEV_BASE_URL
# local      : offline emulation of Jev's Noul/Choice/Score with local cross-encoders
# auto       : typesafe if TYPESAFE_API_KEY, else openrouter if OPENROUTER_API_KEY, else local
JEV_BACKEND = os.getenv("JEV_BACKEND", "auto")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "http://127.0.0.1:8000")     # local Jev-compatible server (ollajev)
JEV_SERVER_KEY = os.getenv("JEV_SERVER_KEY", "local")                 # only needed if the server sets OLLAJEV_API_KEY
AI_GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY", "")      # Vercel AI Gateway (serves Jev, $5/month free credit)

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
LOCAL_NLI_MODEL = os.getenv("LOCAL_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
LOCAL_RERANK_MODEL = os.getenv("LOCAL_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DISCLAIMER = "This is general information, not legal advice."


def resolved_jev_backend() -> str:
    b = JEV_BACKEND.lower()
    if b != "auto":
        return b
    if TYPESAFE_API_KEY:
        return "typesafe"
    if AI_GATEWAY_API_KEY:
        return "vercel"
    if OPENROUTER_API_KEY:
        return "openrouter"
    return "local"
