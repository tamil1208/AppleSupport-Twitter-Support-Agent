"""Two baselines the real system has to beat (assignment requirement)."""
import re

# Fit these from the golden set's label distribution at eval time, not
# hardcoded, so the baseline is honest about what "majority class" means
# for whatever slice it's run on.


def trivial_majority(gold_intents: list, gold_decisions: list):
    """Predict the single most common intent/decision for every example,
    no matter the input. The absolute floor any real system must beat."""
    from collections import Counter

    maj_intent = Counter(gold_intents).most_common(1)[0][0]
    maj_decision = Counter(gold_decisions).most_common(1)[0][0]

    def predict(text: str):
        return maj_intent, maj_decision

    return predict


def simple_keyword(text: str):
    """A simple, unordered, two-keyword baseline -- what someone would
    write in 10 minutes without looking closely at the data. Contrast
    with src/intents.py's 7-rule cascade derived from actually reading
    the traffic."""
    t = text.lower()
    if "battery" in t:
        intent = "battery_drain"
    elif "update" in t:
        intent = "general_update_complaint"
    else:
        intent = "other"
    decision = "escalate" if "fuck" in t else "auto_handle"
    return intent, decision
