"""Decision layer built on Jev, TypeSafe AI's "System One" model.

Jev does not generate text. It reads a state and answers typed questions:
  Noul   - is this statement true?            -> probability of yes (0-1)
  Choice - which one of these options?        -> option key + probabilities + confidence
  Score  - where on this ordered rubric?      -> 0..n-1 (fractional) + probabilities + confidence

Backends
  typesafe   : official API (https://api.typesafe.ai), needs TYPESAFE_API_KEY
  openrouter : Jev served through OpenRouter's System One endpoint, needs OPENROUTER_API_KEY
  vercel     : Jev served through Vercel AI Gateway's TypeSafe-compatible API, needs AI_GATEWAY_API_KEY
  server     : an open-weight Jev-compatible decision model served locally (e.g. ollajev) at JEV_BASE_URL
  local      : offline emulation with small local cross-encoders (NLI + reranker), so the
               project runs without any key. It mimics Jev's interface and output types, not
               its accuracy. Results produced with it are labelled "jev-local" everywhere.

Each question may carry a `hint` used ONLY by the local emulator to know which part of
the state is the premise/passage. Jev and Jev-compatible servers read the whole state and ignore hints.
"""
import asyncio
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache

from .. import config, metrics


# ---------------- question / answer types ----------------

@dataclass
class Noul:
    instructions: str
    hint: dict = field(default_factory=dict)


@dataclass
class Choice:
    instructions: str
    criteria: dict                       # key -> description (or None)
    hint: dict = field(default_factory=dict)


@dataclass
class Score:
    instructions: str
    criteria: list                       # ordered levels, low -> high
    hint: dict = field(default_factory=dict)


@dataclass
class Answer:
    kind: str                            # noul | choice | score
    value: object                        # float | str | float
    probabilities: dict = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def noul(self):
        return float(self.value)

    @property
    def choice(self):
        return self.value

    @property
    def score(self):
        return float(self.value)


# ---------------- engine ----------------

class DecisionEngine:
    def __init__(self, backend=None):
        self.backend = backend or config.resolved_jev_backend()
        if self.backend not in ("typesafe", "openrouter", "vercel", "server", "local"):
            raise ValueError(f"Unknown JEV_BACKEND '{self.backend}'")
        if self.backend == "local":
            self.label = "jev-local (offline emulation)"
        elif self.backend == "server":
            self.label = f"{config.JEV_MODEL.split('/')[-1]} via local Jev-compatible server"
        else:
            self.label = f"Jev via {self.backend}"

    # one request: one state, many questions
    def decide(self, state, questions: dict) -> dict:
        return self.decide_many([(state, questions)])[0]

    # several independent requests (run concurrently against the API, batched locally)
    def decide_many(self, requests: list) -> list:
        if not requests:
            return []
        if self.backend == "local":
            return _local().run(requests)
        return asyncio.run(self._remote_many(requests))

    # ---- remote (TypeSafe SDK) ----
    async def _remote_many(self, requests):
        from typesafe_sdk import AsyncTypeSafeClient
        kwargs = {"timeout": 60}
        model = config.JEV_MODEL
        if self.backend == "typesafe":
            kwargs.update(api_key=config.TYPESAFE_API_KEY)
        elif self.backend == "server":
            kwargs.update(api_key=config.JEV_SERVER_KEY, base_url=config.JEV_BASE_URL.rstrip("/"), timeout=180)
        elif self.backend == "vercel":
            kwargs.update(api_key=config.AI_GATEWAY_API_KEY, base_url="https://ai-gateway.vercel.sh/typesafe")
            if model == "jev-latest":
                model = "typesafe-ai/jev"
        else:
            kwargs.update(api_key=config.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api")
            if model == "jev-latest":
                model = "typesafe/jev-1.13"
        async with AsyncTypeSafeClient(model=model, **kwargs) as client:
            sem = asyncio.Semaphore(8)

            async def one(state, questions):
                async with sem:
                    t0 = time.time()
                    resp = await client.system_one(state=state, questions=_to_sdk(questions))
                    u = getattr(resp, "usage", None)
                    tokens = ((getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)) if u else 0
                    metrics.record_decision(time.time() - t0, len(questions), tokens)
                    return _from_sdk(resp, questions)

            return await asyncio.gather(*(one(s, q) for s, q in requests))


def _to_sdk(questions):
    from typesafe_sdk import Choice as C, Noul as N, Score as S
    out = {}
    for name, q in questions.items():
        if isinstance(q, Noul):
            out[name] = N(instructions=q.instructions)
        elif isinstance(q, Choice):
            out[name] = C(instructions=q.instructions, criteria=q.criteria)
        else:
            out[name] = S(instructions=q.instructions, criteria=list(q.criteria))
    return out


def _from_sdk(resp, questions):
    out = {}
    for name, q in questions.items():
        a = resp.answers[name]
        if isinstance(q, Noul):
            out[name] = Answer("noul", float(a.noul))
        elif isinstance(q, Choice):
            out[name] = Answer("choice", a.choice, dict(a.probabilities or {}), float(a.confidence or 0))
        else:
            out[name] = Answer("score", float(a.score), dict(a.probabilities or {}), float(a.confidence or 0))
    return out


# ---------------- local emulation ----------------

NONE_LOGIT = -3.0        # reranker logit given to a "none of these" option in local Choice questions


def _softmax(xs, temperature=1.0):
    m = max(xs)
    e = [math.exp((x - m) / temperature) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _confidence(probs):
    """Jev reports confidence as 1 - normalised entropy."""
    if len(probs) < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, 1 - h / math.log(len(probs)))


def _flatten(state):
    if isinstance(state, str):
        return state
    return "\n".join(f"{k}: {v}" for k, v in state.items() if v)


class _LocalEngine:
    """Noul/Choice via a zero-shot NLI cross-encoder or a passage reranker, Score via the reranker."""

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.nli = CrossEncoder(config.LOCAL_NLI_MODEL, device="cpu")
        self.rerank = CrossEncoder(config.LOCAL_RERANK_MODEL, device="cpu")
        labels = {v.lower(): int(k) for k, v in (self.nli.model.config.id2label or {}).items()}
        self.ent = labels.get("entailment", 1)
        self.con = labels.get("contradiction", 0)

    def _entail(self, pairs):
        """P(entailment) for (premise, hypothesis) pairs; long premises use their best 1200-char window."""
        expanded, owner = [], []
        for i, (p, h) in enumerate(pairs):
            windows = [p[j:j + 1200] for j in range(0, max(len(p), 1), 1000)] or [""]
            for w in windows[:6]:
                expanded.append((w, h))
                owner.append(i)
        logits = self.nli.predict(expanded, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
        best = [0.0] * len(pairs)
        raw = [-1e9] * len(pairs)
        for i, lg in zip(owner, logits):
            pr = _softmax(list(lg))
            if pr[self.ent] > best[i]:
                best[i] = pr[self.ent]
            raw[i] = max(raw[i], float(lg[self.ent]))
        return best, raw

    def _rerank_logits(self, query, passages):
        idx = [i for i, p in enumerate(passages) if p is not None]
        scores = self.rerank.predict([(query, passages[i][:2000]) for i in idx], show_progress_bar=False) if idx else []
        out = [NONE_LOGIT] * len(passages)
        for i, s in zip(idx, scores):
            out[i] = float(s)
        return out

    def run(self, requests):
        results = []
        for state, questions in requests:
            t0 = time.time()
            text = _flatten(state)
            out = {}
            for name, q in questions.items():
                if isinstance(q, Noul):
                    if "value" in q.hint:                              # derived from other local decisions
                        out[name] = Answer("noul", round(float(q.hint["value"]), 4))
                    elif "query" in q.hint and "passage" in q.hint:   # topical relevance -> reranker
                        logit = self._rerank_logits(q.hint["query"], [q.hint["passage"]])[0]
                        out[name] = Answer("noul", round(1 / (1 + math.exp(-(logit + 1.0) / 1.5)), 4))
                    else:
                        premise = q.hint.get("premise", text)
                        hyp = q.hint.get("hypothesis", q.instructions)
                        p, _ = self._entail([(premise, hyp)])
                        out[name] = Answer("noul", round(p[0], 4))
                elif isinstance(q, Choice):
                    keys = list(q.criteria)
                    if "query" in q.hint and "passages" in q.hint:   # pick the most relevant passage -> reranker
                        raw = self._rerank_logits(q.hint["query"], q.hint["passages"])
                        probs = _softmax(raw, temperature=2.0)
                    else:                                             # zero-shot NLI over option hypotheses
                        premise = q.hint.get("premise", text)
                        hyps = q.hint.get("hypotheses") or [
                            f"{q.instructions.rstrip('?. ')}: {q.criteria[k] or k.replace('_', ' ')}." for k in keys]
                        _, raw = self._entail([(premise, h) for h in hyps])
                        probs = _softmax(raw)
                    best = keys[max(range(len(keys)), key=lambda i: probs[i])]
                    out[name] = Answer("choice", best, dict(zip(keys, map(lambda x: round(x, 4), probs))),
                                       round(_confidence(probs), 4))
                else:  # Score
                    n = len(q.criteria)
                    query = q.hint.get("query")
                    passage = q.hint.get("passage")
                    if query is not None and passage is not None:
                        logit = self._rerank_logits(query, [passage])[0]
                        rel = 1 / (1 + math.exp(-(logit + 1.0) / 1.2))       # 0..1, calibrated on ms-marco logits
                    else:
                        p, _ = self._entail([(text, q.instructions)])
                        rel = p[0]
                    centre = rel * (n - 1)
                    weights = [math.exp(-((i - centre) ** 2) / 0.5) for i in range(n)]
                    s = sum(weights)
                    probs = [w / s for w in weights]
                    out[name] = Answer("score", round(sum(i * p for i, p in enumerate(probs)), 3),
                                       {lvl: round(p, 4) for lvl, p in zip(q.criteria, probs)},
                                       round(_confidence(probs), 4))
            metrics.record_decision(time.time() - t0, len(questions))
            results.append(out)
        return results


@lru_cache(maxsize=1)
def _local():
    return _LocalEngine()


@lru_cache(maxsize=1)
def get_engine():
    return DecisionEngine()
