# Decision log

Non-obvious calls I made while building this, and why. Bullets, roughly
in the order they came up.

1. **Used a 100-row sample instead of the full Kaggle dataset.** Not
   really a choice — Kaggle isn't reachable from the build environment.
   I flagged this loudly in the README and report instead of presenting
   17 examples as if they were 150+. Everything's built to take a full
   CSV path as a drop-in config change, so this isn't structural.

2. **Picked AppleSupport over Tesco / SpotifyCares / VirginTrains.** It
   had the most public-reply threads (13) in the sample, which meant the
   most material to build a taxonomy from and ground replies on. Tesco
   and SpotifyCares had 8 each but noticeably less variety in what
   customers were actually complaining about on a quick read.

3. **Derived the intent taxonomy by reading every AppleSupport message
   first, instead of starting from generic support categories.** The
   result — battery / performance / crash / connectivity /
   feature-conflict / account / general-complaint — reflects a real
   event in the data (the iOS 11 update), not a textbook list. Tradeoff:
   it might not generalize outside that event window, which is worth
   testing next.

4. **Rule order in the classifier goes specific-to-generic on purpose**
   (account, battery, connectivity, crash, feature-conflict, performance,
   then a fallback catching bare "update" mentions). A message matching
   more than one keyword gets the more specific label — "battery" wins
   over a bare "update" mention. That ordering is a real decision, not
   an accident of how the dict happened to get typed.

5. **Redefined "auto-handle" partway through**, once it was clear 13/13
   real AppleSupport replies funnel to DM. I'd originally read it as "no
   human touches this at all." It now means the agent can send the
   public triage reply without a human reviewing that specific reply
   first — see report.md section 1. That changes what the escalation
   metric is actually claiming, and I think it's a more honest claim
   than the one I started with.

6. **Escalation is a short, explicit rule list, not a learned
   classifier.** With 17 labeled examples there's nothing to train an
   escalation model on, and a short auditable rule list (profanity, low
   classifier confidence, PII-touching intent, unknown intent) is
   frankly just more appropriate for a decision with real downside if
   it's wrong.

7. **Retrieval is TF-IDF/cosine, written from scratch, not embeddings.**
   With ~13 historical threads, an embedding index adds dependency
   weight and non-determinism (which model, cached where) for no
   accuracy gain I could actually measure at this n. The interface
   (`top_k(query) -> [Thread]`) doesn't change if this gets swapped for
   a real embedding index later.

8. **Every LLM-touching piece — classifier, reply generator, judge — has
   a deterministic non-LLM fallback, and uses it by default.** The brief
   says the code won't be run on the full dataset and needs to reproduce
   in under 15 minutes, so a hard dependency on an API key that might
   not exist at review time would break that. The LLM path is real and
   wired up (`ANTHROPIC_API_KEY`), just not required for the headline
   numbers.

9. **The golden set is a census of the sample, not a random sample of
   it.** All 17 usable AppleSupport-directed messages, full stop. I
   documented this as a limitation instead of dressing it up as random
   sampling from a pool that doesn't exist here.

10. **The trivial baseline's "majority class" is computed from the
    golden set at eval time, not hardcoded**, so it stays honest if the
    golden set changes — e.g. once real data shows up — instead of
    silently going stale.

11. **The reply-quality rubric is 4 binary checks, not a 1–5 scale.**
    With one labeler and 17 examples, a Likert scale would just be false
    precision. Binary checks are easier to score consistently and easier
    to audit disagreement on — which is exactly what section 4 of the
    report ends up using them for.

12. **The human-agreement check uses one labeler — me — for both the
    "human" column and the rubric design, not an independent rater.**
    I've called this out directly in report.md section 4 rather than
    dressing it up as a real inter-rater study. An actual second rater
    is the first thing I'd add given more time or more people.

13. **`build_unanswered_inbound_for_brand` is separate from the golden
    set** — it's inbound traffic with no public reply in this slice,
    used to show `pipeline.py` running on genuinely "live" input rather
    than only on pre-labeled examples (`python3 src/pipeline.py`). It's
    illustrative, not part of the scored eval.
