"""Check that your Jev key works: sends ONE real request to TypeSafe and prints the typed answers.

  python scripts/check_jev.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from legal_agent import config  # noqa: E402

backend = config.resolved_jev_backend()
if backend == "local":
    sys.exit("No Jev key found. Add AI_GATEWAY_API_KEY=... (Vercel), TYPESAFE_API_KEY=... or OPENROUTER_API_KEY=... "
             "to .env, then run this again.")

from legal_agent.jev import Choice, DecisionEngine, Noul, Score  # noqa: E402

eng = DecisionEngine(backend)
print(f"Calling {eng.label} (model: {config.JEV_MODEL}) ...")
t0 = time.time()
try:
    a = eng.decide(
        {"question": "Can my employer stop me from joining a competitor for 2 years after I resign?"},
        {
            "topic": Choice("Which area of employment law is this about?",
                            {"non_compete": "restrictions after leaving a job", "wages": "salary and deductions",
                             "gratuity": "gratuity and retirement benefits"}),
            "is_legal": Noul("The question is about employment law."),
            "urgency": Score("How urgent does this sound?", ["not urgent", "somewhat", "very urgent"]),
        })
except Exception as e:  # noqa: BLE001
    sys.exit(f"Jev call failed: {type(e).__name__}: {e}")

print(f"OK in {time.time() - t0:.2f}s")
print("  topic   :", a["topic"].choice, a["topic"].probabilities)
print("  is_legal:", round(a["is_legal"].noul, 3))
print("  urgency :", round(a["urgency"].score, 2), "of 2")
