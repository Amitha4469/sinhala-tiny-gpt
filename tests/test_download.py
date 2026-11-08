"""Tests for the NSINA slice logic, using small fake data (no download needed)."""

import json
import random

from sgpt.data.download import (
    article_id,
    iter_records,
    load_jsonl,
    make_slice,
    save_jsonl,
    summarize,
)


def fake_rows(n=200):
    rows = []
    for i in range(n):
        rows.append(
            {
                "Source": "siteA" if i < n // 2 else "siteB",  # stored outlet by outlet, like NSINA
                "Category": "News",
                "Timestamp": "2020",
                "Headline": f"ශීර්ෂය {i}",
                "News Content": "" if i % 10 == 0 else "ශ්‍රී ලංකාව " * (10 + i % 7),
                "URL": f"https://example.lk/{i}",
            }
        )
    return rows


def test_article_id_is_stable_and_unique():
    assert article_id("u", "h") == article_id("u", "h")
    assert article_id("u1", "h") != article_id("u2", "h")
    assert len(article_id("u", "h")) == 16


def test_slice_is_reproducible_and_skips_empty_text():
    a, seen = make_slice(fake_rows(), target_mb=0.01, seed=1)
    b, _ = make_slice(fake_rows(), target_mb=0.01, seed=1)
    assert seen == 200
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert all(r["text"].strip() for r in a)


def test_slice_does_not_depend_on_file_order():
    rows = fake_rows()
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    a, _ = make_slice(rows, target_mb=0.01, seed=1)
    b, _ = make_slice(shuffled, target_mb=0.01, seed=1)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_different_seed_gives_different_slice():
    a, _ = make_slice(fake_rows(), target_mb=0.01, seed=1)
    b, _ = make_slice(fake_rows(), target_mb=0.01, seed=2)
    assert {r["id"] for r in a} != {r["id"] for r in b}


def test_slice_reaches_target_but_not_much_more():
    target_mb = 0.01
    records, _ = make_slice(fake_rows(), target_mb=target_mb, seed=1)
    sizes = [len(r["text"].encode("utf-8")) for r in records]
    target = target_mb * 1024 * 1024
    assert sum(sizes) >= target
    assert sum(sizes) - max(sizes) < target  # minimal: dropping any one would go below target


def test_slice_mixes_sources():
    records, _ = make_slice(fake_rows(), target_mb=0.01, seed=1)
    assert len(summarize(records)["by_source"]) == 2


def test_iter_records_reads_json_array(tmp_path):
    rows = fake_rows(5)
    path = tmp_path / "a.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    assert list(iter_records(str(path))) == rows


def test_iter_records_reads_json_lines_and_concatenated_objects(tmp_path):
    rows = fake_rows(5)
    p1 = tmp_path / "a.jsonl"
    p1.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    p2 = tmp_path / "b.json"
    p2.write_text("".join(json.dumps(r, ensure_ascii=False, indent=2) for r in rows), encoding="utf-8")
    assert list(iter_records(str(p1))) == rows
    assert list(iter_records(str(p2))) == rows


def test_jsonl_round_trip_keeps_zwj(tmp_path):
    records, _ = make_slice(fake_rows(), target_mb=0.005, seed=2)
    path = str(tmp_path / "slice.jsonl")
    save_jsonl(records, path)
    back = load_jsonl(path)
    assert back == records
    assert "\u200d" in back[0]["text"]  # ZWJ in ශ්‍රී survives saving
