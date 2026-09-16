"""
Draft a reply grounded in how the brand has historically handled similar
messages. Uses the top-k retrieved (customer msg -> brand reply) pairs as
few-shot evidence.

Two modes:
- template mode (default, no API key): picks the closest-matching
  historical reply's structural pattern (empathy opener + diagnostic
  question + DM funnel) and slots in intent-specific details. Fully
  deterministic.
- llm mode (ANTHROPIC_API_KEY set): gives Claude the retrieved examples
  as few-shot context and asks it to draft a new reply in the brand's
  voice, addressing the specific intent.
"""
import hashlib
import os
import re


def _stable_hash(text: str) -> int:
    """Python's built-in hash() is randomized per-process (PYTHONHASHSEED),
    which silently broke reproducibility here -- same input picked a
    different empathy opener on different runs. Use a stable hash so
    run_demo.py is actually deterministic run to run."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)

# What follow-up info actually matters per intent, mined from what
# AppleSupport asks for across the historical replies.
_DIAGNOSTIC_ASK = {
    "battery_drain": "which iOS version you're on and roughly how fast the battery is draining",
    "performance_lag": "which iOS version you're on and which apps feel slow",
    "app_crash_freeze": "which app(s) are crashing and your iOS version",
    "connectivity_issue": "which iOS version you're on and whether it happens on WiFi, cellular, or both",
    "feature_conflict": "the exact apps/features involved and what happens when you use them together",
    "account_appstore": "the email on your Apple ID (not the password) so we can look into the code issue",
    "general_update_complaint": "what specifically changed for you since the update, and your iOS version",
    "other": "a bit more detail about what's going on",
}

_EMPATHY_OPENERS = [
    "We know how frustrating that is.",
    "Thanks for flagging this — we're here to help.",
    "We hear you, and we want to get this sorted.",
]


def draft_reply_template(text: str, intent: str, retrieved: list) -> dict:
    opener = _EMPATHY_OPENERS[_stable_hash(text) % len(_EMPATHY_OPENERS)]
    ask = _DIAGNOSTIC_ASK.get(intent, _DIAGNOSTIC_ASK["other"])
    reply = f"{opener} Could you DM us {ask}? We'll take it from there."
    grounding = [
        {"similarity": round(score, 3), "historical_reply": th.brand_reply.text}
        for score, th in retrieved
    ]
    return {"reply": reply, "method": "template", "grounded_on": grounding}


def draft_reply_llm(text: str, intent: str, retrieved: list) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    examples = "\n".join(
        f'- Customer: "{th.customer_msg.text}"\n  Brand replied: "{th.brand_reply.text}"'
        for _, th in retrieved
    )
    prompt = (
        "You are drafting a public Twitter reply for AppleSupport. Match their "
        "actual historical tone and structure shown below: brief empathy, a "
        "specific diagnostic question, and a request to continue in DM. Keep it "
        "under 280 characters. Do not invent policy details not shown in the examples.\n\n"
        f"Historical examples (most similar past cases):\n{examples}\n\n"
        f'New customer message (intent: {intent}): "{text}"\n\n'
        "Reply:"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    reply = resp.content[0].text.strip()
    grounding = [
        {"similarity": round(score, 3), "historical_reply": th.brand_reply.text}
        for score, th in retrieved
    ]
    return {"reply": reply, "method": "llm", "grounded_on": grounding}


def draft_reply(text: str, intent: str, retrieved: list) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return draft_reply_llm(text, intent, retrieved)
        except Exception:
            pass
    return draft_reply_template(text, intent, retrieved)
