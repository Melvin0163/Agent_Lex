"""LexAgent web app: a local law agent for Indian employment law.

Run from the project folder:   python webapp\\server.py      (or double-click run_app.bat)
Then open http://127.0.0.1:8600
"""
import hmac
import os
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.responses import FileResponse, JSONResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

from legal_agent import config, llm, service  # noqa: E402
from legal_agent.jev import get_engine  # noqa: E402

STATIC = Path(__file__).parent / "static"
HOST, PORT = "127.0.0.1", 8600
ALLOWED = {".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm", ".rtf"}
REVIEW_Q = "Review this document clause by clause and flag anything that may conflict with Indian employment law."
DOCS = {}          # id -> prepared document (single local user)
_lock = threading.Lock()   # the local models answer one request at a time


def doc_info(i, d):
    return {"id": i, "name": d.name, "clauses": getattr(d, "n_clauses", 0), "chars": len(d.text)}


async def index(request):
    return FileResponse(STATIC / "index.html")


def _decision_ok():
    eng = get_engine()
    if eng.backend != "server":
        return True, eng.label
    import httpx
    try:
        r = httpx.get(config.JEV_BASE_URL.rstrip("/") + "/v1/models", timeout=3)
        return r.status_code == 200, eng.label
    except Exception:  # noqa: BLE001
        return False, f"ollajev serve {config.JEV_MODEL}"


async def status(request):
    ok_llm, info_llm = await run_in_threadpool(llm.is_available)
    ok_dec, info_dec = await run_in_threadpool(_decision_ok)
    return JSONResponse({"writer": {"ok": bool(ok_llm), "info": str(info_llm)},
                         "decisions": {"ok": ok_dec, "info": info_dec}})


async def samples(request):
    names = sorted(p.name for p in config.SAMPLES_DIR.glob("*") if p.suffix.lower() in (".pdf", ".docx", ".txt"))
    return JSONResponse(names)


def _prepare(path, name):
    doc, _ = service.prepare_document(path, name)
    i = uuid.uuid4().hex[:8]
    DOCS[i] = doc
    return doc_info(i, doc)


async def upload(request):
    form = await request.form()
    f = form.get("file")
    if f is None or Path(f.filename).suffix.lower() not in ALLOWED:
        return JSONResponse({"error": "Use a PDF, Word, text, Markdown, HTML or RTF file."}, status_code=400)
    data = await f.read()
    if len(data) > 50 * 1024 * 1024:
        return JSONResponse({"error": "The file is larger than 50 MB."}, status_code=400)
    try:
        path = service.save_upload(f.filename, data)
        return JSONResponse(await run_in_threadpool(_prepare, path, f.filename))
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Couldn't read {f.filename}: {e}"}, status_code=400)


async def add_sample(request):
    name = Path(request.path_params["name"]).name
    p = config.SAMPLES_DIR / name
    if not p.exists():
        return JSONResponse({"error": "No such sample."}, status_code=404)
    return JSONResponse(await run_in_threadpool(_prepare, p, name))


async def remove_doc(request):
    DOCS.pop(request.path_params["id"], None)
    return JSONResponse({"ok": True})


def _ask(engine, question, docs):
    with _lock:
        t0 = time.time()
        r = service.ask(engine, question, docs)
    return {
        "engine": engine, "route": r.route, "answer": r.answer, "gated": r.gated,
        "issues": list(r.issues or []),
        "sources": [{"kind": s.get("kind"), "label": s.get("label"), "title": s.get("title"),
                     "status": s.get("status"), "text": s.get("text", "")} for s in r.sources],
        "metrics": r.metrics, "seconds": round(time.time() - t0, 1),
        "steps": [{"step": s.get("node"), "secs": s.get("secs"), "info": str(s.get("info", ""))} for s in r.steps],
        "files_read": list(getattr(r, "okf_path", []) or []),
    }


async def ask(request):
    body = await request.json()
    engine = body.get("engine") if body.get("engine") in service.PIPELINES else "jev_okf"
    question = (body.get("question") or "").strip()
    if body.get("review"):
        question = REVIEW_Q
    if not question:
        return JSONResponse({"error": "Type a question first."}, status_code=400)
    docs = [DOCS[i] for i in body.get("docs", []) if i in DOCS]
    try:
        return JSONResponse(await run_in_threadpool(_ask, engine, question, docs))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "Connect" in type(e).__name__ or "connect" in msg.lower():
            msg = ("A local model isn't reachable. Check that Ollama is running, and for the JEV + OKF engine, "
                   f"that the decision model is running (ollajev serve {config.JEV_MODEL}).")
        return JSONResponse({"error": msg}, status_code=500)


ACCESS_CODE = os.environ.get("LEXAGENT_ACCESS_CODE", "").strip()


class AccessCode(BaseHTTPMiddleware):
    """When LEXAGENT_ACCESS_CODE is set (for a shared link), every /api request must carry it."""

    async def dispatch(self, request, call_next):
        if ACCESS_CODE and request.url.path.startswith("/api/"):
            given = request.headers.get("x-access-code", "")
            if not hmac.compare_digest(given.encode(), ACCESS_CODE.encode()):
                return JSONResponse({"error": "Enter the access code to use this app.", "needs_code": True}, status_code=401)
        return await call_next(request)


app = Starlette(middleware=[Middleware(AccessCode)], routes=[
    Route("/", index),
    Route("/api/status", status),
    Route("/api/samples", samples),
    Route("/api/samples/{name}", add_sample, methods=["POST"]),
    Route("/api/upload", upload, methods=["POST"]),
    Route("/api/docs/{id}", remove_doc, methods=["DELETE"]),
    Route("/api/ask", ask, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])

if __name__ == "__main__":
    print("Preparing the knowledge bundle and models (first run takes a minute)...")
    service.ensure_bundle()
    from legal_agent.rag.retriever import warm_up
    warm_up()
    url = f"http://{HOST}:{PORT}"
    if ACCESS_CODE:
        print("Access code is ON: visitors must enter it before using the app.")
    print(f"LexAgent is running at {url}  (press Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
