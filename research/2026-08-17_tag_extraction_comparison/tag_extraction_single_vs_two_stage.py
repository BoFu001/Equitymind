"""
Single-stage vs two-stage industry tag extraction.

Two-stage (production, Research Logs 02/04/05):
    10-K Item 1 -> gpt-4o 150-200 word overview -> LLM_MODEL_LIGHT tag extraction

Single-stage (this experiment):
    10-K Item 1 -> LLM_MODEL tag extraction directly

Both consume the SAME business_text, so any difference is attributable to
the intermediate representation, not to different input.

Motivation: NVDA's confirmed extraction gap (Research Log 05 3.3) was
localised to stage one — its production overview contains the word
"semiconductor" zero times, so the light extraction model never had the
chance to tag it. This tests whether that loss is systematic.

Expected asymmetry, NOT a hypothesis that one method wins:
  - two-stage under-extracts (recall gap; the 200-word budget forces the
    model to spend words on products and market position, not category)
  - single-stage may over-extract (precision gap; an open-ended judgement
    over noisy input is the Research Log 01 risk profile, e.g. tagging
    AAPL as semiconductor from parametric knowledge)

Verdicts are NOT assigned here — every disagreement must be checked
against the 10-K text before being classified (Research Log 04 4.3).
"""

import sys
import json
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL
from extract_industry_tags import extract_tags_llm


def _load_module(path: Path, name: str):
    """scripts/ is not a package, so load by path rather than import."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_uso = _load_module(ROOT / "scripts" / "update_stock_overviews.py", "uso")
fetch_business_text = _uso.fetch_business_text
generate_overview = _uso.generate_overview

client = OpenAI(api_key=OPENAI_API_KEY)

# Diagnostic sample. Purpose-selected to expose failure modes; NOT a
# random sample and therefore cannot support any rate estimate.
DIAGNOSTIC = [
    "NVDA",                                    # confirmed gap, Log 05 3.3
    "AAPL", "SNOW",                            # confirmed NON-gaps, Log 04 4.3
    "AMZN", "GOOGL", "BRK-B", "HON",           # multi-business
    "WAT", "ZTS", "CME", "AZO", "SBUX",        # narrow business
]

# Deliberately mirrors the classification clause of OVERVIEW_PROMPT so the
# comparison isolates architecture, not prompt wording.
SINGLE_STAGE_PROMPT = """Based on the following official 10-K business \
description, list this company's industry/theme classification tags \
(multiple applicable tags, e.g. technology, e-commerce, cloud computing).

Tags must be grounded in the 10-K text below. Do not introduce outside \
knowledge. Tag what this company IS or DOES, not what it purchases or \
depends on.

10-K Business Description:
{business_text}

Return ONLY a comma-separated list of lowercase tags, nothing else:"""


def extract_tags_single_stage(business_text: str) -> list[str]:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{
            "role": "user",
            "content": SINGLE_STAGE_PROMPT.format(business_text=business_text),
        }],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


def main():
    results = []

    for i, ticker in enumerate(DIAGNOSTIC, 1):
        print(f"[{i}/{len(DIAGNOSTIC)}] {ticker}...", flush=True)

        business_text, filing_date = fetch_business_text(ticker, None)
        if business_text is None:
            results.append({"ticker": ticker, "status": "skipped"})
            continue

        overview = generate_overview(business_text)
        two_stage = extract_tags_llm(overview)
        single_stage = extract_tags_single_stage(business_text)

        only_single = sorted(set(single_stage) - set(two_stage))
        only_two = sorted(set(two_stage) - set(single_stage))

        results.append({
            "ticker": ticker,
            "status": "ok",
            "filing_date": filing_date,
            "business_text_chars": len(business_text),
            "overview": overview,
            "two_stage_tags": two_stage,
            "single_stage_tags": single_stage,
            "only_single_stage": only_single,
            "only_two_stage": only_two,
        })

        print(f"    two-stage:    {two_stage}", flush=True)
        print(f"    single-stage: {single_stage}", flush=True)
        print(f"    only single:  {only_single}", flush=True)
        print(f"    only two:     {only_two}", flush=True)
        print(flush=True)

    out = HERE / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
