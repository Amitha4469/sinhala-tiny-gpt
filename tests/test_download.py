"""Tests for the NSINA slice logic, using a small fake dataset (no download needed)."""

from sgpt.data.download import article_id, load_jsonl, make_slice, save_jsonl, summarize


def fake_dataset(n=100):
    rows = []
    for i in range(n):
        rows.append(
            {
                "Source": "siteA" if i < n // 2 else "siteB",  # stored outlet by outlet, like NSINA
                "Category": "News",
                "Timestamp": "2020",
                "Headline": f"ශීර්ෂය {i}",
                "News Content": "" if i % 10 == 0 else "ශ්‍රී ලංකාව " * 20,
                "URL": f"https://example.lk/{i}",
            }
        )
    return rows


def test_article_id_is_stable_and_unique():
    assert article_id("u", "h") == article_id("u", "h")
    assert article_id("u1", "h") != article_id("u2", "h")
    assert len(article_id("u", "h")) == 16


def test_slice_is_reproducible_and_skips_empty_text():
    ds = fake_dataset()
    a = make_slice(ds, target_mb=0.01, seed=1)
    b = make_slice(ds, target_mb=0.01, seed=1)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert all(r["text"].strip() for r in a)


def test_slice_mixes_sources():
    # Random order must not just take the first outlet
    ds = fake_dataset()
    s = summarize(make_slice(ds, target_mb=0.01, seed=1))
    assert len(s["by_source"]) == 2


def test_jsonl_round_trip_keeps_zwj(tmp_path):
    records = make_slice(fake_dataset(), target_mb=0.005, seed=2)
    path = str(tmp_path / "slice.jsonl")
    save_jsonl(records, path)
    back = load_jsonl(path)
    assert back == records
    assert "\u200d" in back[0]["text"]  # ZWJ in ශ්‍රී survives saving
