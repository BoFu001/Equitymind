# RQ1 results

60 runs: 30 questions x 2 runs. Group N = 24 real user questions, Group S = 6 stress questions.
Rules 2 and 3 judged by script on all 60 runs; rule 4 on all 14 multi-company reports;
rule 1 by hand on a sample of 15 reports (seed 42).

| Rule | Group | Judged | Pass | Fail | Undecidable numbers | Not applicable / not sampled | Runs failed | Questions failed | 95% CI (runs) |
|---|---|---|---|---|---|---|---|---|---|
| Invented number | N | 10 | 10 | 0 | 0 | 38 | 0/10 | 0/8 | 0% to 28% |
| Invented number | S | 5 | 5 | 0 | 0 | 7 | 0/5 | 0/4 | 0% to 43% |
| Unsupported best | N | 0 | 0 | 0 | 0 | 48 | 0/0 | 0/0 | - |
| Unsupported best | S | 6 | 6 | 0 | 0 | 6 | 0/6 | 0/3 | 0% to 39% |
| Wrong order | N | 0 | 0 | 0 | 0 | 48 | 0/0 | 0/0 | - |
| Wrong order | S | 4 | 4 | 0 | 0 | 8 | 0/4 | 0/2 | 0% to 49% |
| Number on wrong company | N | 6 | 6 | 0 | 15 | 42 | 0/6 | 0/3 | 0% to 39% |
| Number on wrong company | S | 8 | 8 | 0 | 12 | 4 | 0/8 | 0/4 | 0% to 32% |

## Run-to-run agreement
Questions whose two runs got different verdicts on any rule:
none

## Reading the table
- 'Judged' counts runs with a pass or fail verdict. 'Not applicable' runs had no report,
  or the rule did not apply (companies named by the user; a single-company report).
  For 'Invented number' the column is runs not in the 15-run sample.
- 'Undecidable numbers' are numbers that appear in more than one company's data block; they are
  counted, not judged. They never add to fails.
- Group S rates are provoked, not real-use rates.
