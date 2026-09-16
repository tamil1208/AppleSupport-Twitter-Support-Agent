"""
LLM-as-judge for reply quality, with a heuristic fallback so the harness
is runnable offline (no API key) in under 15 minutes.

Rubric (4 binary checks, matches what a human reviewer would actually
check for this brand -- see report.md "what good looks like"):
  1. grounded         -- structurally matches the brand's real pattern
                          (empathy + specific ask + DM handoff), not a
                          generic canned line
  2. on_topic          -- the diagnostic question asked is actually
                          relevant to the symptom the customer named
  3. not_redundant     -- doesn't ask for info the customer already gave
  4. tone_appropriate  -- doesn't apply a complaint-handling tone to a
                          non-complaint (e.g. praise) message

Composite score = checks passed / 4.

Human agreement: data/golden/human_reply_scores.csv holds the same 4
checks hand-scored by a human reader (see decision_log.md #12 on why
this substitutes for a full inter-rater study at n=17). agreement()
compares the two.
"""
import csv
import os
import re


def heuristic_judge(customer_text: str, intent: str, reply: str) -> dict:
    t = customer_text.lower()
    r = reply.lower()

    grounded = bool(re.search(r"\bdm\b", r)) and any(
        p in r for p in ("we know", "we hear you", "thanks for flagging", "we'd like", "happy to help")
    )

    # on_topic: does the reply's ask reference something plausibly tied to
    # the intent (crude keyword check against the same map reply_gen uses)
    topic_terms = {
        "battery_drain": ["battery", "drain"],
        "performance_lag": ["ios version", "apps feel slow", "slow"],
        "app_crash_freeze": ["crash", "ios version"],
        "connectivity_issue": ["wifi", "cellular", "ios version"],
        "feature_conflict": ["apps/features", "together"],
        "account_appstore": ["apple id", "code"],
        "general_update_complaint": ["update", "ios version"],
        "other": ["detail"],
        "praise_feedback": ["detail"],  # no dedicated ask -- expected to fail
    }
    terms = topic_terms.get(intent, [])
    on_topic = any(term in r for term in terms)

    # not_redundant: flag if the reply asks for iOS version/details the
    # customer already explicitly stated
    already_gave_version = bool(re.search(r"\b(latest|ios ?1\d|iphone ?\d)\b", t))
    asks_for_version = "ios version" in r
    not_redundant = not (already_gave_version and asks_for_version)

    # tone_appropriate: flag empathy/apology language applied to clearly
    # positive customer text
    positive_signal = bool(re.search(r"\b(love|great|super help|problem solved|thanks so much)\b", t))
    apology_signal = bool(re.search(r"\b(frustrating|sorted|we hear you|we know)\b", r))
    tone_appropriate = not (positive_signal and apology_signal)

    checks = {
        "grounded": grounded,
        "on_topic": on_topic,
        "not_redundant": not_redundant,
        "tone_appropriate": tone_appropriate,
    }
    return {**checks, "score": sum(checks.values()) / 4}


def llm_judge(customer_text: str, intent: str, reply: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()
    prompt = f"""Score this customer support reply against 4 yes/no checks.
Customer message: "{customer_text}"
Classified intent: {intent}
Draft reply: "{reply}"

Checks:
1. grounded: does it follow a real support-reply structure (acknowledge + specific ask + handoff), not a generic non-answer?
2. on_topic: is the question asked actually relevant to what the customer described?
3. not_redundant: does it avoid re-asking for info the customer already gave?
4. tone_appropriate: does the tone match the customer's actual sentiment (don't apologize to a happy customer)?

Respond as JSON only: {{"grounded": bool, "on_topic": bool, "not_redundant": bool, "tone_appropriate": bool}}"""
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    import json

    checks = json.loads(resp.content[0].text.strip())
    return {**checks, "score": sum(checks.values()) / 4}


def judge(customer_text: str, intent: str, reply: str) -> dict:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return llm_judge(customer_text, intent, reply)
        except Exception:
            pass
    return heuristic_judge(customer_text, intent, reply)


def load_human_scores(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["tweet_id"]: row for row in csv.DictReader(f)}


def agreement(judge_results: dict, human_scores: dict) -> dict:
    """Per-check agreement rate between automated judge and human labels,
    over the examples both have scores for."""
    checks = ["grounded", "on_topic", "not_redundant", "tone_appropriate"]
    common_ids = set(judge_results) & set(human_scores)
    out = {"n": len(common_ids)}
    for check in checks:
        matches = 0
        for tid in common_ids:
            j = str(judge_results[tid][check])
            h = human_scores[tid][check]
            if j.lower() == h.lower():
                matches += 1
        out[check] = round(matches / len(common_ids), 2) if common_ids else 0.0
    return out
