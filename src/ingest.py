"""
Ingest the Customer Support on Twitter CSV and reconstruct conversation
threads (customer message -> brand reply -> customer follow-up -> ...).

Works on any subsample of the Kaggle dataset with the standard schema:
tweet_id, author_id, inbound, created_at, text, response_tweet_id,
in_response_to_tweet_id
"""
import csv
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Tweet:
    tweet_id: str
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_id: Optional[str]
    in_response_to_tweet_id: Optional[str]


@dataclass
class Thread:
    """A single customer message paired with the brand's reply to it,
    plus the shallow conversational context around it."""
    thread_id: str
    brand: str
    customer_msg: Tweet
    brand_reply: Optional[Tweet]
    prior_customer_turns: list = field(default_factory=list)  # earlier msgs in same convo


def clean_text(text: str) -> str:
    """Strip @handles used purely for routing and collapse whitespace.
    Keeps the text otherwise intact -- we want real customer language for
    intent classification, not a sanitized version."""
    t = re.sub(r"\s+", " ", text).strip()
    return t


def load_tweets(csv_path: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tweets = {}
        for row in reader:
            tid = row["tweet_id"]
            tweets[tid] = Tweet(
                tweet_id=tid,
                author_id=row["author_id"],
                inbound=(row["inbound"].strip().lower() == "true"),
                created_at=row["created_at"],
                text=row["text"],
                response_tweet_id=row["response_tweet_id"] or None,
                in_response_to_tweet_id=row["in_response_to_tweet_id"] or None,
            )
        return tweets


def build_threads_for_brand(tweets: dict, brand_handle: str) -> list:
    """Every (customer message -> this brand's reply) pair, in the public
    timeline. Doesn't require the full conversation to resolve -- most of
    resolution for these brands actually happens in DM (see report), so we
    treat the public reply as the unit of grounding evidence."""
    threads = []
    for t in tweets.values():
        if t.author_id == brand_handle and not t.inbound:
            parent_id = t.in_response_to_tweet_id
            parent = tweets.get(parent_id) if parent_id else None
            if parent is None or not parent.inbound:
                continue
            # walk further back for context, capped at 2 hops
            prior = []
            cur = parent
            hops = 0
            while cur.in_response_to_tweet_id and hops < 2:
                gp = tweets.get(cur.in_response_to_tweet_id)
                if gp is None:
                    break
                prior.append(gp)
                cur = gp
                hops += 1
            threads.append(
                Thread(
                    thread_id=f"{parent.tweet_id}->{t.tweet_id}",
                    brand=brand_handle,
                    customer_msg=parent,
                    brand_reply=t,
                    prior_customer_turns=list(reversed(prior)),
                )
            )
    return threads


def build_unanswered_inbound_for_brand(tweets: dict, brand_handle: str) -> list:
    """Customer messages @-ing the brand that never got a public reply in
    this slice of data -- used as 'live' inbound traffic to run the agent
    on, since threads-with-replies are needed for grounding, not for eval
    input."""
    # crude: any inbound tweet mentioning the brand handle that isn't the
    # parent of any brand reply in this slice
    replied_parents = set()
    for t in tweets.values():
        if t.author_id == brand_handle and not t.inbound and t.in_response_to_tweet_id:
            replied_parents.add(t.in_response_to_tweet_id)
    out = []
    handle_pat = re.compile(rf"@{re.escape(brand_handle)}\b", re.IGNORECASE)
    for t in tweets.values():
        if t.inbound and handle_pat.search(t.text) and t.tweet_id not in replied_parents:
            out.append(t)
    return out


if __name__ == "__main__":
    tweets = load_tweets("data/raw/sample.csv")
    threads = build_threads_for_brand(tweets, "AppleSupport")
    print(f"loaded {len(tweets)} tweets, {len(threads)} AppleSupport threads")
    for th in threads[:3]:
        print("-", th.customer_msg.text[:80], "=>", th.brand_reply.text[:80])
