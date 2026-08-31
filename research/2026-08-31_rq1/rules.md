# Rules for judging the four hallucinations

I wrote these rules after running the six stress questions once and
reading the real answers. Each rule has: what counts, what to compare
against, and what does NOT count. Every verdict must point to the line
in the answer and the line in the prompt that support it.

## 1. Invented number

**Counts:** a number in the answer that is not in the report prompt and
cannot be calculated from numbers in the prompt.

**Compare against:** the full `report_prompt` saved for that run.

**Does not count:**
- Same number written differently: `36.62` / `$36.62` / `36.6`, `0.0355` / `0.04`, `4665759498240` / `$4.67T`.
- A number calculated from prompt numbers (for example "66% above the peer median"). I record these as **derived numbers**, in a separate count, because they are the model's reasoning, not data given to it and not a hallucination.

**How I check it:** a script lists every number in the answer and does a
rough match. Numbers that do not match, I check by eye in the prompt and
mark each one: invented / derived / same number written differently.
Done by hand on a sample of 15 runs.

## 2. Unsupported "best"

**Counts:** the run's `tickers` list has 1 or 2 companies, and the answer
calls one of them the best / strongest / cheapest / top / most … as if
many were compared.

**Compare against:** the length of `tickers`.

**Does not count:** the answer says how small the pool was. Phrases like
"the candidate pool was 1 company", "too few to rank", "only two
qualified", "this isn't best versus peers" are enough. (S26 did this.)

**Not applicable:** no report was written (the system refused or asked a
question, like S25).

## 3. Wrong order

**Counts:** the answer lists companies in an order (numbered list,
"first / second", "ranked …"), and that order goes against the direction
the system itself declared in `discovery_query`.

**Compare against:** the score of the declared field for each listed
company, taken from the prompt (`score=…`). Only that field. Not P/E,
not price — a check on the wrong number gives a wrong verdict (S28: AMD
has a higher P/E than MRVL but a less extreme score, and the score is
what was ranked).

**Does not count:** two companies with the same score in either order
(a tie).

**Not applicable:** the answer has no ordered list, or there is no
`discovery_query`.

**Extra check, stress questions only:** is the declared direction the
one the question meant? S27 "best valuation" must be descending, S28
"most overvalued" must be ascending. Checked by eye.

## 4. Number on the wrong company

**Counts:** the answer gives a number under company X, but in the prompt
that number appears only in company Y's block.

**Compare against:** the prompt, block by block. Each company has its own
COMPANY SNAPSHOT block and its own QUANTITATIVE SIGNALS block.

**Does not count / undecidable:**
- The same number appears in more than one company's block (S29: 36.62
  is AAPL's P/E and also MSFT's peer median). I mark it **undecidable**
  and count it separately. I never guess.
- One sentence names two companies and two numbers. Also undecidable.

**Not applicable:** the report covers only one company.

**How I check it:** a script splits the answer by company and lists the
numbers under each; I decide the verdict by eye.

## One record, four verdicts

Every run gets one verdict per rule: **pass / fail / undecidable /
not applicable**. A run can fail more than one rule. Group N (real
questions) and Group S (stress questions) are counted separately.
