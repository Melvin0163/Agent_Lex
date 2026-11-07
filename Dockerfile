# LexAgent web app (Starlette + both pipelines). The LLM runs in a separate Ollama container
# (see docker-compose.yml); the decision model defaults to the offline stand-in (JEV_BACKEND=local).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: about 2 GB smaller than the default CUDA build, and enough for e5-small
# and the two cross-encoders. requirements.txt then sees torch as already installed.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY . .

# Download the embedding model and the local decision-model cross-encoders at build time,
# so the container answers its first question without fetching anything.
RUN python -c "from legal_agent import config; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(config.EMBED_MODEL); \
CrossEncoder(config.LOCAL_NLI_MODEL); CrossEncoder(config.LOCAL_RERANK_MODEL)"

ENV APP_HOST=0.0.0.0 \
    APP_PORT=8600 \
    LLM_PROVIDER=ollama \
    OLLAMA_URL=http://ollama:11434/v1 \
    OLLAMA_MODEL=qwen2.5:7b \
    JEV_BACKEND=local

RUN useradd --create-home app && chown -R app:app /app
USER app

EXPOSE 8600
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8600/', timeout=4)" || exit 1

CMD ["python", "webapp/server.py", "--no-browser"]
