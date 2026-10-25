# Evaluation: Agentic RAG vs JEV + OKF (retrieval mode)

18 questions | LLM: ollama:qwen2.5:7b | Decision model: laya-typed-decisions via local Jev-compatible server | 2026-10-25 17:36

| Metric | Agentic RAG | JEV + OKF |
|---|---|---|
| Correct routing (%) | 72.2 | 94.4 |
| Expected section retrieved (%) | 91.7 | 75.0 |
| Law sources from expected Act (%) | 65.0 | 64.7 |
| Avg law sections sent to LLM | 6.67 | 3.61 |
| Avg seconds per question | 6.36 | 10.58 |
| Avg LLM calls per question | 1.89 | 0.0 |
| Avg LLM tokens per question | 366.06 | 0.0 |
| Avg Jev decision calls per question | 0.0 | 7.22 |

## Per question

| id | question | expected | Agentic RAG route / hit / secs | JEV + OKF route / hit / secs |
|---|---|---|---|---|
| q01 | Can my company enforce a non-compete after I resign? | doc_question; Indian Contract Act 27 | law_question / True / 42.14 | doc_question / False / 14.17 |
| q02 | Is the Rs 4 lakh bond in my agreement valid? | doc_question; Indian Contract Act 74 | doc_question / True / 5.06 | doc_question / False / 13.11 |
| q03 | How soon must my final settlement be paid after I resign? | doc_question; Code on Wages 17 | doc_question / True / 5.06 | doc_question / True / 11.93 |
| q04 | Can the company deduct any amount from my salary? | doc_question; Code on Wages 18 | doc_question / True / 4.2 | doc_question / True / 13.17 |
| q05 | I am on a fixed-term contract, am I eligible for gratuity? | doc_question; Code on Social Security Act 53 | law_question / True / 4.46 | doc_question / True / 12.03 |
| q06 | Is 12 weeks of maternity leave enough under the law? | doc_question; Code on Social Security Act verify | law_question / True / 4.53 | law_question / True / 7.45 |
| q07 | Does our company need an Internal Committee for sexual harassment? | doc_question; Sexual Harassment of Women at Workplace (Prevention 4 | law_question / False / 3.77 | doc_question / True / 10.59 |
| q08 | Is the 7-day complaint window in our policy valid? | doc_question; Sexual Harassment of Women at Workplace (Prevention 9 | doc_question / True / 5.3 | doc_question / False / 8.56 |
| q09 | Was my termination legal? | doc_question; -  | law_question / None / 4.62 | doc_question / None / 9.83 |
| q10 | Review this contract and tell me what to be careful about | review; -  | review / None / 5.32 | review / None / 14.9 |
| q11 | Review this agreement | review; -  | review / None / 6.09 | review / None / 18.06 |
| q12 | Am I eligible for gratuity after 4 years? | law_question; Code on Social Security Act 53 | law_question / True / 3.91 | law_question / True / 9.16 |
| q13 | What is the maternity leave duration? | law_question; Code on Social Security Act verify | law_question / True / 3.84 | law_question / True / 11.63 |
| q14 | How long does the company have to pay my dues after I leave? | law_question; Code on Wages 17 | law_question / True / 3.89 | law_question / True / 13.16 |
| q15 | Can I be forced to work overtime without extra pay? | law_question; Code on Wages verify | law_question / True / 4.39 | law_question / True / 11.77 |
| q16 | What is the best biryani in Chennai? | out_of_scope; -  | out_of_scope / None / 2.68 | out_of_scope / None / 1.56 |
| q17 | Should I sue my employer? | law_question; -  | law_question / None / 2.54 | law_question / None / 8.92 |
| q18 | Write me a Python script to sort a list | out_of_scope; -  | out_of_scope / None / 2.59 | out_of_scope / None / 0.49 |

> Decisions were made by the offline Jev emulator (local cross-encoders), not the hosted Jev model. Set TYPESAFE_API_KEY (or OPENROUTER_API_KEY) in .env and re-run for real Jev results.
