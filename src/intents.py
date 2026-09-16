"""
Intent taxonomy for AppleSupport, derived by reading the actual inbound
tweets in the sample (see decision_log.md #3). Six intents cover the
observed traffic; "other" is the overflow bucket.

Classifier: keyword/regex rules by default (zero-cost, deterministic,
fully reproducible without an API key -- see decision_log.md #8). If
ANTHROPIC_API_KEY is set, classify() will instead call Claude with the
same taxonomy for higher recall on phrasing the rules miss, and falls
back to the rule-based label if the call fails.
"""
import os
import re

INTENTS = {
    "battery_drain": "Complaint about battery draining faster than expected, usually tied to an iOS update.",
    "performance_lag": "Phone/apps slow, laggy, or taking longer to load since an update.",
    "app_crash_freeze": "Apps crashing, freezing, or the phone becoming unresponsive.",
    "connectivity_issue": "WiFi, Bluetooth, or cellular connectivity dropping or misbehaving.",
    "feature_conflict": "A specific feature or combination of apps not working as expected together.",
    "account_appstore": "App Store / Apple ID account issues -- codes, purchases, sign-in.",
    "general_update_complaint": "Frustration directed at an update with no specific diagnosable symptom named.",
    "other": "Doesn't fit the above.",
}

_RULES = [
    ("account_appstore", re.compile(r"\b(code|i-?store|app store|apple id|redeem|sign ?in|purchase)\b", re.I)),
    ("battery_drain", re.compile(r"\bbattery\b", re.I)),
    ("connectivity_issue", re.compile(r"\b(wifi|wi-fi|bluetooth|disconnect|signal|cellular)\b", re.I)),
    ("app_crash_freeze", re.compile(r"\b(crash\w*|freez\w*|frozen|unresponsive|stuck)\b", re.I)),
    ("feature_conflict", re.compile(r"\b(at the same time|while (using|listening)|can'?t (use|do) .* (and|while))\b", re.I)),
    ("performance_lag", re.compile(r"\b(slow|lag|takes? ages|loading|sluggish)\b", re.I)),
    ("general_update_complaint", re.compile(r"\b(update|ios ?1\d)\b.*\b(sucks?|hate|horrible|disgrace|ruin|paraly[sz]ed|fix (it|this))\b", re.I)),
]


def classify_rule_based(text: str) -> tuple:
    """Returns (intent, matched_rule_or_None). First matching rule wins;
    order reflects specificity (see decision_log.md #4)."""
    for intent, pattern in _RULES:
        if pattern.search(text):
            return intent, pattern.pattern
    if re.search(r"\bupdate\b", text, re.I):
        return "general_update_complaint", "fallback:update"
    return "other", None


def classify_llm(text: str) -> str:
    """Optional Claude-based classifier. Requires ANTHROPIC_API_KEY."""
    import anthropic  # local import so the module works without the package

    client = anthropic.Anthropic()
    taxonomy = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    prompt = (
        "Classify this customer support tweet into exactly one intent label "
        f"from this taxonomy:\n{taxonomy}\n\n"
        f'Tweet: "{text}"\n\n'
        "Respond with only the label, nothing else."
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    label = resp.content[0].text.strip().lower()
    return label if label in INTENTS else "other"


def classify(text: str) -> dict:
    """Public entrypoint. Returns dict with intent + method + confidence
    proxy (rule match = high confidence, fallback = low)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            intent = classify_llm(text)
            return {"intent": intent, "method": "llm", "confidence": "high"}
        except Exception as e:  # noqa: BLE001
            pass  # fall through to rules
    intent, matched = classify_rule_based(text)
    confidence = "high" if matched and not matched.startswith("fallback") else "low"
    return {"intent": intent, "method": "rule", "confidence": confidence, "matched": matched}


if __name__ == "__main__":
    tests = [
        "battery runs out in half the time since the update",
        "my apps keep crashing and phone freezes every 5 minutes",
        "wifi disconnects frequently since I updated",
        "the new update doesn't let me listen to music and use whatsapp at the same time",
        "I need a new code for my I-store, msg is too many sent",
        "fix this update. it's horrible",
        "everything takes ages to load now",
    ]
    for t in tests:
        print(classify(t), "|", t[:60])
