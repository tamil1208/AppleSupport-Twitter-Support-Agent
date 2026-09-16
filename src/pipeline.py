"""End-to-end: raw message -> intent -> grounded reply -> escalation decision."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ingest import load_tweets, build_threads_for_brand, build_unanswered_inbound_for_brand
from intents import classify
from retrieval import TfidfIndex
from reply_gen import draft_reply
from escalation import decide

BRAND = "AppleSupport"


class SupportAgent:
    def __init__(self, csv_path: str, brand: str = BRAND):
        self.brand = brand
        self.tweets = load_tweets(csv_path)
        self.threads = build_threads_for_brand(self.tweets, brand)
        self.index = TfidfIndex(self.threads)

    def handle(self, text: str, k: int = 3) -> dict:
        cls = classify(text)
        retrieved = self.index.top_k(text, k=k)
        reply = draft_reply(text, cls["intent"], retrieved)
        esc = decide(text, cls["intent"], cls["confidence"])
        return {
            "input": text,
            "intent": cls["intent"],
            "intent_method": cls["method"],
            "intent_confidence": cls["confidence"],
            "reply": reply["reply"],
            "reply_method": reply["method"],
            "grounded_on": reply["grounded_on"],
            "decision": esc["decision"],
            "decision_reasons": esc["reasons"],
        }


if __name__ == "__main__":
    agent = SupportAgent("data/raw/sample.csv")
    unanswered = build_unanswered_inbound_for_brand(agent.tweets, BRAND)
    print(f"{len(unanswered)} unanswered inbound AppleSupport messages in this slice\n")
    for t in unanswered[:5]:
        out = agent.handle(t.text)
        print("IN: ", t.text[:90])
        print("  intent:", out["intent"], f"({out['intent_confidence']})")
        print("  decision:", out["decision"], "-", out["decision_reasons"][0])
        print("  reply:", out["reply"])
        print()
