# Locked plan (pre-registration) – 8 Nov 2026

Written **before** any training (Jira: STGWKS-11). Do not change it after seeing results. If something must change, add a dated note at the bottom explaining why.

## 1. Research question

For a tiny GPT built from scratch, trained with the same model size and the same amount of text, how does the choice of tokenizer (character, grapheme, byte-level BPE, grapheme-based BPE) affect how well it predicts Sinhala text and how often it writes broken Sinhala?

## 2. Hypotheses

- **H1:** Grapheme-based tokenizers (grapheme, GPE) reach lower bits per character than byte-level BPE.
- **H2:** Byte-level BPE and character models produce more broken conjuncts than grapheme-based models. Grapheme and GPE can almost never break a letter by design, so the real comparison is character vs byte-BPE. This is stated openly in the report.
- **H3 (exploratory):** The ranking by bits per character matches the ranking by human rating.

## 3. Metrics

### Bits per character (main)

- Total loss in nats ÷ ln(2) ÷ number of characters in the original test text
- Character = Unicode code point after NFC, including ZWJ, spaces and punctuation
- Each article evaluated separately, starting with a start-of-text token
- Sliding window evaluation: stride = half the context, only the new half is scored

### Broken conjunct rate (new metric)

- Invalid grapheme clusters ÷ all grapheme clusters in generated text
- Invalid = vowel sign or virama with no consonant before it, two vowel signs in a row, double virama, or ZWJ not between a virama and a consonant
- 200 samples × ~500 characters per model, temperature 1.0
- Validated first on real test text (should be close to 0)

### Human rating (supporting)

- 50 samples per tokenizer, temperature 0.8, shuffled, labels hidden
- Scored 1–5 on fluency, spelling and sense
- Single rater for now; a second native reader is added if one becomes available. Otherwise noted as a limitation.

## 4. Fairness controls

| Control | Rule |
| --- | --- |
| Model size | Same non-embedding parameters for all tokenizers |
| Training amount | Same number of characters seen, not steps or tokens |
| Settings | Same optimizer, LR schedule and batch size in characters |
| Randomness | 3 seeds per tokenizer |
| Test data | Same locked test articles for all models |

Exact numbers (model size, characters seen) are set after STGWKS-16 (corpus stats) and STGWKS-32 (speed benchmark).

## 5. When a difference counts

A difference counts as real only if the gap between averages is larger than twice the spread across seeds.

## 6. Success levels

| Level | What it means |
| --- | --- |
| Minimum | From-scratch GPT verified against PyTorch + fair bits-per-character comparison of all four tokenizers |
| Full | Minimum + broken conjunct rate + blind human rating |
| Stretch | Rubber curing PINN |

## Change log

_(none yet)_
