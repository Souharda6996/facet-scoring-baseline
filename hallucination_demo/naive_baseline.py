"""
Minimal "naive" single-shot scorer -- deliberately built the way an
unsophisticated first pass often looks: no taxonomy gate, no abstention
option, no anchors, just "rate this facet 1-5 given this conversation."

This exists ONLY as a contrast baseline for hallucination_demo/examples.md.
It is intentionally the wrong way to do this -- run it side-by-side with
src/pipeline.py's taxonomy-gated output to demonstrate the difference
concretely, not to suggest this is a usable design.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llm_client import call_llm, LLMCallError  # noqa: E402

NAIVE_SYSTEM_PROMPT = (
    "You are a personality/behavior analyst. Given a conversation and a facet name, "
    "rate the conversation on that facet from 1 (low) to 5 (high). Always give your best "
    "estimate even if evidence is limited -- the user needs a number."
)

NAIVE_USER_TEMPLATE = """Conversation:
\"\"\"
{conversation}
\"\"\"

Facet to rate: "{facet}"

Respond with JSON: {{"score": <1-5>, "reason": "<short reason>"}}
"""


def naive_score(conversation: str, facet: str, model: str = None) -> dict:
    user_prompt = NAIVE_USER_TEMPLATE.format(conversation=conversation.strip(), facet=facet)
    kwargs = {"model": model} if model else {}
    try:
        raw = call_llm(NAIVE_SYSTEM_PROMPT, user_prompt, **kwargs)
        return json.loads(raw)
    except (LLMCallError, json.JSONDecodeError) as e:
        return {"score": None, "reason": f"naive baseline call/parse failed: {e}"}


if __name__ == "__main__":
    cases = [
        ("I've been trying to get pregnant for a year now and my doctor mentioned something "
         "about my hormone levels being off, but honestly I don't remember the exact numbers "
         "she said.", "FSH level"),
        ("I toss and turn every night, wake up gasping sometimes, and I'm exhausted all day no "
         "matter how early I sleep. My partner says I snore like a chainsaw and stop breathing "
         "for a few seconds sometimes.", "Sleep Apnea"),
        ("I just finished reading a dense 900-page philosophy book on phenomenology and wrote a "
         "20-page analysis comparing Husserl and Heidegger over the weekend, purely for fun. I "
         "love diving into ideas like that.", "Intelligence Quotient (IQ)"),
    ]
    for conv, facet in cases:
        result = naive_score(conv, facet)
        print(f"\nFacet: {facet}\nNaive result: {result}")
