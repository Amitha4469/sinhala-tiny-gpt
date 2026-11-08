"""Download NSINA and save a reproducible random slice (Jira: STGWKS-13).

NSINA (sinhala-nlp/NSINA, CC-BY-SA-4.0) has ~500K Sinhala news articles from
10 outlets, stored as one ~1.9 GB JSON file. Loading it with the `datasets`
library parses the whole file into memory and crashes Colab (out of RAM).

So we:
1. download the raw JSON file with huggingface_hub,
2. read it **one article at a time** with ijson (streaming, low memory),
3. keep a seeded random slice of ~40 MB while streaming.

How the random slice works (without loading everything):
every article gets a random-looking key = hash(seed + article id). We keep the
articles with the *smallest* keys until their text reaches the target size.
This equals "shuffle all articles with a fixed seed, then take from the top",
but needs only ~40 MB of memory. The result does not depend on file order, so
it mixes all outlets and is reproducible.

Each saved record:
    id         stable 16-char hash of URL + headline (NSINA has no ID field)
    source     news outlet (kept so news bias can be discussed later)
    category, timestamp, headline, text, url
"""

import hashlib
import heapq
import json
import os
from collections import Counter

REPO_ID = "sinhala-nlp/NSINA"
TEXT_FIELD = "News Content"


def article_id(url, headline):
    """Stable ID for an article: same input always gives the same ID."""
    key = f"{url or ''}|{headline or ''}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def download_nsina(token=None):
    """Download the raw NSINA JSON file(s). Returns a list of local file paths."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [
        f
        for f in list_repo_files(REPO_ID, repo_type="dataset", token=token)
        if f.endswith((".json", ".jsonl"))
    ]
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found in {REPO_ID}")
    return [hf_hub_download(REPO_ID, f, repo_type="dataset", token=token) for f in files]


def _open_json(path):
    """Open a JSON file in binary mode, skip a UTF-8 BOM, and return (file, first_char)."""
    f = open(path, "rb")
    if f.read(3) != b"\xef\xbb\xbf":
        f.seek(0)
    start = f.tell()
    ch = b" "
    while ch and ch.isspace():
        ch = f.read(1)
    f.seek(start)
    return f, ch


def iter_records(path):
    """Yield one article (dict) at a time from a JSON array, JSON Lines, or concatenated JSON objects."""
    import ijson

    f, first = _open_json(path)
    with f:
        if first == b"[":
            yield from ijson.items(f, "item")
        else:
            yield from ijson.items(f, "", multiple_values=True)


def _sort_key(seed, rid):
    return int(hashlib.sha1(f"{seed}|{rid}".encode("utf-8")).hexdigest()[:16], 16)


def make_slice(rows, target_mb=40, seed=1234, progress_every=None):
    """Keep a seeded random slice of `rows` whose text adds up to about `target_mb` MB (UTF-8).

    `rows` can be any iterable of NSINA-style dicts (streamed or a list).
    Articles with empty text are skipped.
    Returns (records, rows_seen). Records are ordered by their random key.
    """
    target = int(target_mb * 1024 * 1024)
    heap = []  # max-heap by key: entries are (-key, tiebreak, size, record)
    total = 0
    seen = 0

    for row in rows:
        seen += 1
        if progress_every and seen % progress_every == 0:
            print(f"  read {seen:,} articles, kept {len(heap):,}")

        text = row.get(TEXT_FIELD) or ""
        if not text.strip():
            continue

        rid = article_id(row.get("URL"), row.get("Headline"))
        key = _sort_key(seed, rid)

        # Already have enough text and this key is larger than all kept ones: it would be dropped anyway.
        if heap and total >= target and key > -heap[0][0]:
            continue

        size = len(text.encode("utf-8"))
        record = {
            "id": rid,
            "source": row.get("Source"),
            "category": row.get("Category"),
            "timestamp": row.get("Timestamp"),
            "headline": row.get("Headline"),
            "text": text,
            "url": row.get("URL"),
        }
        heapq.heappush(heap, (-key, seen, size, record))
        total += size

        # Drop the largest keys while we would still be at or above target without them.
        while heap and total - heap[0][2] >= target:
            _, _, s, _ = heapq.heappop(heap)
            total -= s

    records = [r for _, _, _, r in sorted(heap, key=lambda e: -e[0])]
    return records, seen


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
