"""Download NSINA and save a reproducible random slice (Jira: STGWKS-13).

NSINA (sinhala-nlp/NSINA, CC-BY-SA-4.0) has ~507K Sinhala news articles from
10 outlets. The dataset is stored outlet by outlet, so taking the first N rows
would give almost only one outlet. Instead we shuffle all rows with a fixed seed
and take articles in that random order until we reach the target size. This keeps
the natural mix of outlets and makes the slice reproducible.

Each saved record:
    id         stable 16-char hash of URL + headline (NSINA has no ID field)
    source     news outlet (kept so news bias can be discussed later)
    category, timestamp, headline, text, url
"""

import hashlib
import json
import os
from collections import Counter

import numpy as np

TEXT_FIELD = "News Content"


def article_id(url, headline):
    """Stable ID for an article: same input always gives the same ID."""
    key = f"{url or ''}|{headline or ''}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def load_nsina(token=None):
    """Load the full NSINA train split from Hugging Face (needs an accepted, gated access token)."""
    from datasets import load_dataset

    return load_dataset("sinhala-nlp/NSINA", split="train", token=token)


def make_slice(ds, target_mb=40, seed=1234):
    """Take articles in a seeded random order until the text reaches `target_mb` MB (UTF-8).

    `ds` can be a Hugging Face Dataset or any list of dicts with NSINA field names.
    Articles with empty text are skipped. Returns a list of records.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    target_bytes = int(target_mb * 1024 * 1024)

    records, total = [], 0
    for i in order:
        row = ds[int(i)]
        text = row.get(TEXT_FIELD) or ""
        if not text.strip():
            continue
        records.append(
            {
                "id": article_id(row.get("URL"), row.get("Headline")),
                "source": row.get("Source"),
                "category": row.get("Category"),
                "timestamp": row.get("Timestamp"),
                "headline": row.get("Headline"),
                "text": text,
                "url": row.get("URL"),
            }
        )
        total += len(text.encode("utf-8"))
        if total >= target_bytes:
            break
    return records


def summarize(records):
    """Basic numbers about a slice: article count, size, mix of outlets and categories."""
    n_bytes = sum(len(r["text"].encode("utf-8")) for r in records)
    n_chars = sum(len(r["text"]) for r in records)
    return {
        "articles": len(records),
        "text_mb": round(n_bytes / 1024 / 1024, 2),
        "characters": n_chars,
        "by_source": dict(Counter(r["source"] for r in records).most_common()),
        "by_category": dict(Counter(r["category"] for r in records).most_common()),
    }


def save_jsonl(records, path):
    """One JSON record per line, Sinhala kept as real characters (ensure_ascii=False)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
