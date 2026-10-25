# Evaluation: Agentic RAG vs JEV + OKF (retrieval mode)

18 questions | LLM: ollama:qwen2.5:7b | Decision model: laya-typed-decisions via local Jev-compatible server | 2026-10-25 17:42

| Metric | Agentic RAG | JEV + OKF |
|---|---|---|
| Correct routing (%) | 72.2 | 94.4 |
| Expected section retrieved (%) | 83.3 | 91.7 |
| Law sources from expected Act (%) | 57.6 | 67.2 |
| Avg law sections sent to LLM | 6.72 | 3.78 |
| Avg seconds per question | 5.87 | 9.75 |
| Avg LLM calls per question | 1.89 | 0.0 |
| Avg LLM tokens per question | 366.39 | 0.0 |
| Avg Jev decision calls per question | 0.0 | 8.0 |

## Per question

| id | question | expected | Agentic RAG route / hit / secs | JEV + OKF route / hit / secs |
|---|---|---|---|---|
| q01 | Can my company enforce a non-compete after I resign? | doc_question; Indian Contract Act 27 | law_question / True / 37.81 | doc_question / True / 13.53 |
| q02 | Is the Rs 4 lakh bond in my agreement valid? | doc_question; Indian Contract Act 74 | doc_question / True / 3.95 | doc_question / True / 13.32 |
| q03 | How soon must my final settlement be paid after I resign? | doc_question; Code on Wages 17 | doc_question / True / 4.8 | doc_question / True / 10.37 |
| q04 | Can the company deduct any amount from my salary? | doc_question; Code on Wages 18 | doc_question / True / 4.28 | doc_question / True / 12.55 |
| q05 | I am on a fixed-term contract, am I eligible for gratuity? | doc_question; Code on Social Security Act 53 | law_question / True / 3.78 | doc_question / True / 12.8 |
| q06 | Is 12 weeks of maternity leave enough under the law? | doc_question; Code on Social Security Act verify | law_question / True / 4.14 | law_question / True / 8.21 |
| q07 | Does our company need an Internal Committee for sexual harassment? | doc_question; Sexual Harassment of Women at Workplace (Prevention 4 | law_question / False / 3.59 | doc_question / True / 12.92 |
| q08 | Is the 7-day complaint window in our policy valid? | doc_question; Sexual Harassment of Women at Workplace (Prevention 9 | doc_question / False / 4.87 | doc_question / False / 10.94 |
| q09 | Was my termination legal? | doc_question; -  | law_question / None / 4.32 | doc_question / None / 11.84 |
| q10 | Review this contract and tell me what to be careful about | review; -  | review / None / 4.75 | review / None / 13.67 |
| q11 | Review this agreement | review; -  | review / None / 5.23 | review / None / 14.91 |
| q12 | Am I eligible for gratuity after 4 years? | law_question; Code on Social Security Act 53 | law_question / True / 3.92 | law_question / True / 8.05 |
| q13 | What is the maternity leave duration? | law_question; Code on Social Security Act verify | law_question / True / 3.89 | law_question / True / 7.92 |
| q14 | How long does the company have to pay my dues after I leave? | law_question; Code on Wages 17 | law_question / True / 4.44 | law_question / True / 8.33 |
| q15 | Can I be forced to work overtime without extra pay? | law_question; Code on Wages verify | law_question / True / 4.39 | law_question / True / 7.89 |
| q16 | What is the best biryani in Chennai? | out_of_scope; -  | out_of_scope / None / 2.71 | out_of_scope / None / 0.51 |
| q17 | Should I sue my employer? | law_question; -  | law_question / None / 2.03 | law_question / None / 7.23 |
| q18 | Write me a Python script to sort a list | out_of_scope; -  | out_of_scope / None / 2.81 | out_of_scope / None / 0.52 |
