"""Per-question tracing used to compare the two engines' cost."""
from legal_agent import metrics


def test_trace_accumulates_llm_and_decision_calls():
    t = metrics.start("rag")
    metrics.record_llm(0.5, 100, 20)
    metrics.record_llm(0.25, 50, None)
    metrics.record_decision(0.1, 3, tokens=40)
    metrics.record_embed()
    s = t.finish().summary()
    assert s["llm_calls"] == 2 and s["llm_tokens"] == 170
    assert s["decision_calls"] == 1 and s["decision_questions"] == 3 and s["decision_tokens"] == 40
    assert s["embed_calls"] == 1


def test_timed_node_records_a_step():
    t = metrics.start("jev_okf")
    node = metrics.timed_node("route", lambda state: {"route": "law_question"}, describe=lambda o: o["route"])
    assert node({}) == {"route": "law_question"}
    assert t.steps[0]["node"] == "route" and t.steps[0]["info"] == "law_question"
