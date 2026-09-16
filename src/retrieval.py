"""
Retrieval over the brand's historical (customer msg -> brand reply) pairs.
Given a new inbound message, find the K most similar past customer
messages and return the brand's actual replies to them as grounding
evidence for the reply generator.

TF-IDF + cosine, not embeddings: with ~13 historical threads for the demo
brand an embedding model is overkill and non-reproducible without a key
(decision_log.md #8). Swap in a real embedding index at scale -- the
interface (top_k(query) -> [Thread]) doesn't change.
"""
import math
import re
from collections import Counter


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    def __init__(self, threads: list):
        self.threads = threads
        self.docs = [tokenize(t.customer_msg.text) for t in threads]
        self.df = Counter()
        for doc in self.docs:
            for term in set(doc):
                self.df[term] += 1
        self.n = len(self.docs)
        self.doc_vecs = [self._vectorize(doc) for doc in self.docs]

    def _vectorize(self, tokens: list) -> dict:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            idf = math.log((1 + self.n) / (1 + self.df.get(term, 0))) + 1
            vec[term] = count * idf
        return vec

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        da = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        db = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return num / (da * db)

    def top_k(self, query_text: str, k: int = 3, exclude_thread_id: str = None) -> list:
        qvec = self._vectorize(tokenize(query_text))
        scored = []
        for th, dvec in zip(self.threads, self.doc_vecs):
            if exclude_thread_id and th.thread_id == exclude_thread_id:
                continue
            score = self._cosine(qvec, dvec)
            scored.append((score, th))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


if __name__ == "__main__":
    from ingest import load_tweets, build_threads_for_brand

    tweets = load_tweets("data/raw/sample.csv")
    threads = build_threads_for_brand(tweets, "AppleSupport")
    idx = TfidfIndex(threads)
    results = idx.top_k("my battery is draining so fast after the update", k=3)
    for score, th in results:
        print(f"{score:.3f}", th.customer_msg.text[:60], "=>", th.brand_reply.text[:60])
