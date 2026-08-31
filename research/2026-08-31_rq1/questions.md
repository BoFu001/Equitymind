# Question set for RQ1

Written before any question was drawn. Drawn by
`sample_questions.py`, seed 42, saved to `questions.json`.
**Frozen once committed. No changes after seeing any result.**

## Four hallucinations measured

- **Invented number** — a number in the report that was not in the data given to the model.
- **Unsupported "best"** — the pool has one or two companies, but the report says "strongest" or "cheapest".
- **Wrong order** — the user asked for cheapest first; the list is in the opposite order.
- **Number on the wrong company** — company A's number written under company B.

## Two pools

- **Pool A — named stocks** (`questions_by_category.txt`, 84 questions, 8 intent patterns). Single-stock path. Can show: invented number; wrong company (comparisons).
- **Pool B — sector questions** (`quora_*_stocks.txt`, 526 lines). Discovery path: pool → rank → report. Can show: unsupported "best"; wrong order; wrong company.

Pool A alone cannot trigger three of the four. Pool B alone misses the single-stock path. So I draw from both.

## Group N — 24 real questions

**Pool A, 10:** one or two per intent pattern (patterns 1 and 3 give two, the rest one). Lines I marked `##########` are excluded (77 candidates left). No two questions about the same stock inside one pattern.

**Pool B, 14:** only questions with a ranking keyword (best, top, cheapest,
undervalued, overvalued, which, what are), because only those can reach the
ranking path where three of the four hallucinations occur. Random draw of
5 / 5 / 4 from the semiconductor / tech / healthcare files.

If the system refuses a question or asks for clarification, I record
"no report" for that run. It counts as neither a failure nor a pass.

## Group S — 6 stress questions, written by me

| ID | Question | Targets | Why it is hard |
|---|---|---|---|
| S25 | What is the strongest HVAC stock? | Unsupported "best" | only one company (CARR) has the tag `hvac` |
| S26 | Which aviation company has the best momentum? | Unsupported "best" | only one company (BA) has the tag `aviation` |
| S27 | Which healthcare companies have the best valuation? | Wrong order | "best" must sort `valuation_score` high to low |
| S28 | Which semiconductor stocks are the most overvalued? | Wrong order | "most overvalued" must sort low to high (the case fixed in Log 06) |
| S29 | Compare the valuation of AAPL, MSFT and GOOGL. | Wrong company | three companies with close valuation numbers |
| S30 | Which is more expensive right now, NVDA, AMD or AVGO? | Wrong company | three companies with close numbers |

No stress question for invented number; it is checked by hand.

## Limitations

- 30 questions, one universe, one model.
- Pool B is limited to ranking-form questions, so its rate is a rate among ranking requests, not among all sector questions.
- Group S questions are written to provoke failures, so their rate is not a real-use rate.
