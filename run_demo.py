#!/usr/bin/env python3
"""
Reproduces the headline results. Runs in well under 15 minutes (no API
key required -- uses the rule-based classifier / template reply generator
by default; set ANTHROPIC_API_KEY to use Claude for classification, reply
drafting, and judging instead).

Usage: python3 run_demo.py
Outputs: outputs/predictions.csv, outputs/eval_summary.md
"""
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "eval"))

from pipeline import SupportAgent
from baselines import trivial_majority, simple_keyword
from metrics import accuracy, confusion, per_class_report
from judge import judge, load_human_scores, agreement


def load_golden(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    golden = load_golden("data/golden/golden_eval.csv")
    agent = SupportAgent("data/raw/sample.csv")

    gold_intents = [row["gold_intent"] for row in golden]
    gold_decisions = [row["gold_decision"] for row in golden]

    system_intents, system_decisions, rows_out = [], [], []
    for row in golden:
        out = agent.handle(row["text"], k=3)
        system_intents.append(out["intent"])
        system_decisions.append(out["decision"])
        rows_out.append({**row, **{f"pred_{k}": v for k, v in out.items()}})

    triv_predict = trivial_majority(gold_intents, gold_decisions)
    triv_intents, triv_decisions = [], []
    for row in golden:
        i, d = triv_predict(row["text"])
        triv_intents.append(i)
        triv_decisions.append(d)

    kw_intents, kw_decisions = [], []
    for row in golden:
        i, d = simple_keyword(row["text"])
        kw_intents.append(i)
        kw_decisions.append(d)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/predictions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_out[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    results = {
        "n": len(golden),
        "system": {
            "intent_acc": accuracy(system_intents, gold_intents),
            "decision_acc": accuracy(system_decisions, gold_decisions),
        },
        "trivial_baseline": {
            "intent_acc": accuracy(triv_intents, gold_intents),
            "decision_acc": accuracy(triv_decisions, gold_decisions),
        },
        "simple_keyword_baseline": {
            "intent_acc": accuracy(kw_intents, gold_intents),
            "decision_acc": accuracy(kw_decisions, gold_decisions),
        },
    }

    intent_report = per_class_report(system_intents, gold_intents)
    intent_confusion = confusion(system_intents, gold_intents)
    decision_confusion = confusion(system_decisions, gold_decisions)

    judge_results = {}
    for row in rows_out:
        judge_results[row["tweet_id"]] = judge(row["text"], row["pred_intent"], row["pred_reply"])
    avg_reply_score = sum(j["score"] for j in judge_results.values()) / len(judge_results)
    human_scores = load_human_scores("data/golden/human_reply_scores.csv")
    judge_agreement = agreement(judge_results, human_scores)

    with open("outputs/eval_summary.md", "w", encoding="utf-8") as f:
        f.write("# Eval summary\n\n")
        f.write(f"n = {results['n']} (golden set)\n\n")
        f.write("| | Intent accuracy | Decision accuracy |\n|---|---|---|\n")
        f.write(f"| System | {results['system']['intent_acc']:.2f} | {results['system']['decision_acc']:.2f} |\n")
        f.write(f"| Trivial baseline (majority class) | {results['trivial_baseline']['intent_acc']:.2f} | {results['trivial_baseline']['decision_acc']:.2f} |\n")
        f.write(f"| Simple keyword baseline | {results['simple_keyword_baseline']['intent_acc']:.2f} | {results['simple_keyword_baseline']['decision_acc']:.2f} |\n\n")
        f.write("## Per-class intent report (system)\n\n")
        f.write("| intent | support | precision | recall | f1 |\n|---|---|---|---|---|\n")
        for label, m in sorted(intent_report.items()):
            f.write(f"| {label} | {m['support']} | {m['precision']} | {m['recall']} | {m['f1']} |\n")
        f.write("\n## Intent confusion (gold -> predicted counts)\n\n")
        for gold, preds in intent_confusion.items():
            f.write(f"- **{gold}**: {preds}\n")
        f.write("\n## Decision confusion (gold -> predicted counts)\n\n")
        for gold, preds in decision_confusion.items():
            f.write(f"- **{gold}**: {preds}\n")
        f.write(f"\n## Reply quality (judge: {'LLM' if os.environ.get('ANTHROPIC_API_KEY') else 'heuristic'})\n\n")
        f.write(f"Average composite score: {avg_reply_score:.2f} / 1.00 (4-check rubric)\n\n")
        f.write("### Agreement with human-scored labels\n\n")
        f.write(f"n = {judge_agreement['n']}\n\n")
        f.write("| check | agreement |\n|---|---|\n")
        for check in ["grounded", "on_topic", "not_redundant", "tone_appropriate"]:
            f.write(f"| {check} | {judge_agreement[check]:.2f} |\n")

    print(open("outputs/eval_summary.md").read())


if __name__ == "__main__":
    main()
