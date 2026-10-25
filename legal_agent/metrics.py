"""Per-question tracing: latency per step, LLM calls/tokens, decision-model calls.

Both pipelines record into the same Trace so their efficiency can be compared fairly.
"""
import contextvars
import time
from dataclasses import dataclass, field

_current = contextvars.ContextVar("trace", default=None)


@dataclass
class Trace:
    pipeline: str
    steps: list = field(default_factory=list)       # [{"node", "secs", "info"}]
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_secs: float = 0.0
    decision_calls: int = 0                          # Jev requests (or local emulated ones)
    decision_questions: int = 0                      # typed questions answered
    decision_tokens: int = 0
    decision_secs: float = 0.0
    embed_calls: int = 0
    started: float = field(default_factory=time.time)
    total_secs: float = 0.0

    def add_step(self, node, secs, info=""):
        self.steps.append({"node": node, "secs": round(secs, 3), "info": info})

    def finish(self):
        self.total_secs = round(time.time() - self.started, 3)
        return self

    def summary(self):
        return {
            "pipeline": self.pipeline,
            "total_secs": self.total_secs,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_prompt_tokens + self.llm_completion_tokens,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "llm_secs": round(self.llm_secs, 2),
            "decision_calls": self.decision_calls,
            "decision_questions": self.decision_questions,
            "decision_tokens": self.decision_tokens,
            "decision_secs": round(self.decision_secs, 2),
            "embed_calls": self.embed_calls,
        }


def start(pipeline: str) -> Trace:
    t = Trace(pipeline)
    _current.set(t)
    return t


def current():
    return _current.get()


def record_llm(secs, prompt_tokens, completion_tokens):
    t = _current.get()
    if t:
        t.llm_calls += 1
        t.llm_secs += secs
        t.llm_prompt_tokens += prompt_tokens or 0
        t.llm_completion_tokens += completion_tokens or 0


def record_decision(secs, n_questions, tokens=0):
    t = _current.get()
    if t:
        t.decision_calls += 1
        t.decision_questions += n_questions
        t.decision_tokens += tokens or 0
        t.decision_secs += secs


def record_embed():
    t = _current.get()
    if t:
        t.embed_calls += 1


def timed_node(name, fn, describe=None):
    """Wrap a LangGraph node so its wall time is recorded in the current trace."""
    def wrapper(state):
        t0 = time.time()
        out = fn(state)
        tr = _current.get()
        if tr:
            info = describe(out) if describe else ""
            tr.add_step(name, time.time() - t0, info)
        return out
    wrapper.__name__ = name
    return wrapper
