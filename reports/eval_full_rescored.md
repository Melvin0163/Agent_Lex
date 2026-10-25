# Evaluation: Agentic RAG vs JEV + OKF (full mode)

18 questions | LLM: ollama:qwen2.5:7b | Decision model: laya-typed-decisions via local Jev-compatible server | 2026-10-25 17:59

> Re-scored on 2026-10-25 18:01 from the saved answers of the run on 2026-10-25 17:59 with the corrected refusal and citation rules; no model was called.

| Metric | Agentic RAG | JEV + OKF |
|---|---|---|
| Correct routing (%) | 72.2 | 94.4 |
| Expected section retrieved (%) | 83.3 | 91.7 |
| Law sources from expected Act (%) | 58.8 | 67.2 |
| Avg law sections sent to LLM | 6.67 | 3.78 |
| Avg seconds per question | 30.96 | 24.41 |
| Avg LLM calls per question | 4.89 | 0.83 |
| Avg LLM tokens per question | 8263.56 | 1589.56 |
| Avg Jev decision calls per question | 0.0 | 10.44 |
| Correct refuse/answer behaviour (%) | 94.4 | 94.4 |
| Answer cites the expected section (%) | 75.0 | 83.3 |
| Valid citations (%) | 93.3 | 95.6 |

## Per question

| id | question | expected | Agentic RAG route / hit / secs / refused / cites expected | JEV + OKF route / hit / secs / refused / cites expected |
|---|---|---|---|---|
| q01 | Can my company enforce a non-compete after I resign? | doc_question; Indian Contract Act 27 | law_question / True / 37.85 / False / True | doc_question / True / 31.89 / False / True |
| q02 | Is the Rs 4 lakh bond in my agreement valid? | doc_question; Indian Contract Act 74 | doc_question / True / 48.5 / False / True | doc_question / True / 32.28 / False / False |
| q03 | How soon must my final settlement be paid after I resign? | doc_question; Code on Wages 17 | doc_question / True / 24.18 / False / True | doc_question / True / 21.4 / False / True |
| q04 | Can the company deduct any amount from my salary? | doc_question; Code on Wages 18 | doc_question / True / 14.03 / False / True | doc_question / True / 28.91 / False / True |
| q05 | I am on a fixed-term contract, am I eligible for gratuity? | doc_question; Code on Social Security Act 53 | law_question / True / 27.97 / False / False | doc_question / True / 30.47 / False / True |
| q06 | Is 12 weeks of maternity leave enough under the law? | doc_question; Code on Social Security Act verify | law_question / True / 24.56 / False / True | law_question / True / 25.95 / False / True |
| q07 | Does our company need an Internal Committee for sexual harassment? | doc_question; Sexual Harassment of Women at Workplace (Prevention 4 | law_question / False / 18.24 / False / False | doc_question / True / 30.04 / False / True |
| q08 | Is the 7-day complaint window in our policy valid? | doc_question; Sexual Harassment of Women at Workplace (Prevention 9 | doc_question / False / 29.17 / False / False | doc_question / False / 25.94 / False / False |
| q09 | Was my termination legal? | doc_question; - ; should refuse | law_question / None / 64.11 / True / None | doc_question / None / 30.0 / False / None |
| q10 | Review this contract and tell me what to be careful about | review; -  | review / None / 53.54 / False / None | review / None / 40.26 / False / None |
| q11 | Review this agreement | review; -  | review / None / 76.18 / False / None | review / None / 47.91 / False / None |
| q12 | Am I eligible for gratuity after 4 years? | law_question; Code on Social Security Act 53 | law_question / True / 32.65 / False / True | law_question / True / 28.51 / False / True |
| q13 | What is the maternity leave duration? | law_question; Code on Social Security Act verify | law_question / True / 31.48 / False / True | law_question / True / 17.2 / False / True |
| q14 | How long does the company have to pay my dues after I leave? | law_question; Code on Wages 17 | law_question / True / 12.99 / False / True | law_question / True / 16.0 / False / True |
| q15 | Can I be forced to work overtime without extra pay? | law_question; Code on Wages verify | law_question / True / 32.52 / False / True | law_question / True / 23.29 / False / True |
| q16 | What is the best biryani in Chennai? | out_of_scope; - ; should refuse | out_of_scope / None / 0.65 / True / None | out_of_scope / None / 0.52 / True / None |
| q17 | Should I sue my employer? | law_question; - ; should refuse | law_question / None / 25.87 / False / None | law_question / None / 8.27 / True / None |
| q18 | Write me a Python script to sort a list | out_of_scope; - ; should refuse | out_of_scope / None / 2.86 / True / None | out_of_scope / None / 0.51 / True / None |
