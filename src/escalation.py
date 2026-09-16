"""
Auto-handle vs escalate-to-human, with a stated reason.

Key finding driving this design (see report.md, problem framing): in this
brand's actual historical data, 13/13 public replies funnel the customer
to DM -- meaning a human agent (or at least a non-public channel) handles
every single case in the sample. There is no observed instance of Apple
fully resolving a technical issue in the public reply itself.

So "auto-handle" here is deliberately narrow: it means the AGENT can send
the public triage reply itself (asking for diagnostic info, same as a
human would type first) without a human reviewing it pre-send. It does
NOT mean the underlying issue is resolved without a human -- see report
for why that distinction matters and what would change it (a real KB of
resolutions, which this dataset doesn't contain).

Escalate (hold for human review before anything is sent) when:
  - profanity / high-intensity language directed at the brand (brand risk)
  - low classifier confidence (agent doesn't know what it's dealing with)
  - account/billing-adjacent intent (irreversible or PII-touching actions)
  - message contains no diagnosable content at all ("other" intent) after
    a templated ask has already been tried once (not detectable from a
    single message in this slice, so treated conservatively as escalate
    on first sight of "other")
"""
import re

_PROFANITY = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|wtf|bullshit)\b", re.I)

ESCALATE_INTENTS = {"account_appstore"}  # touches Apple ID / purchase -> PII, human review


def decide(text: str, intent: str, confidence: str) -> dict:
    reasons = []
    if _PROFANITY.search(text):
        reasons.append("message contains high-intensity/profane language -- brand-risk, human should see it before anything public goes out")
    if confidence == "low":
        reasons.append("intent classifier had low confidence -- agent isn't sure what it's dealing with")
    if intent in ESCALATE_INTENTS:
        reasons.append(f"intent '{intent}' touches account/purchase details -- PII risk, requires human handling")
    if intent == "other":
        reasons.append("message doesn't match any known intent pattern -- no safe template to draw on")

    decision = "escalate" if reasons else "auto_handle"
    if not reasons:
        reasons.append(f"intent '{intent}' is a known, low-risk pattern with a standard triage reply and high classifier confidence")
    return {"decision": decision, "reasons": reasons}


if __name__ == "__main__":
    tests = [
        ("battery runs out in half the time", "battery_drain", "high"),
        ("I need a new code for my I-store", "account_appstore", "high"),
        ("this fucking update ruined my phone", "general_update_complaint", "high"),
        ("weird stuff happening idk", "other", "low"),
    ]
    for text, intent, conf in tests:
        print(decide(text, intent, conf))
