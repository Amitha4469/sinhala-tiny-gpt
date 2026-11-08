# Sinhala Tiny GPT

A small GPT language model built **from scratch in NumPy** (manual forward and backward passes, no autograd), trained on Sinhala text.

This is a practice project before the DA603B Deep Learning course at Kristianstad University. The goal is to learn as much as possible by building every part myself, and to answer one real research question properly.

## Research question

For a tiny GPT built from scratch, trained with the same model size and the same amount of text, how does the choice of tokenizer (character, grapheme, byte-level BPE, grapheme-based BPE) affect how well it predicts Sinhala text and how often it writes broken Sinhala?

Main metric: **bits per character** (fair across tokenizers). New metric: **broken conjunct rate**. The full locked plan is in [`docs/preregistration.md`](docs/preregistration.md).

## Project structure

```
sgpt/
  backend.py      NumPy / CuPy switch (import xp from here, never numpy directly)
  data/           download, cleaning, de-duplication, splits
  tokenization/   character, grapheme, BPE and GPE tokenizers
  nn/             from-scratch layers, loss, optimizers, gradient checker
  models/         bigram, GPT
experiments/      training and evaluation scripts
notebooks/        Colab notebooks
tests/            pytest tests (gradient checks, PyTorch comparisons)
docs/             plan, derivations, notes
```

## Rules for this repo

- Everything in `sgpt/nn` and `sgpt/models` uses only `xp` from `sgpt.backend`. No PyTorch, no autograd.
- PyTorch is used **only** in `tests/` and as a reference implementation for checking results and running extra sweeps.
- Data and checkpoints live on Google Drive, not in git.
- Commit messages: `STGWKS-XX: short imperative description` (Jira project STGWKS).

## Data

**NSINA** – a news corpus for Sinhala (~507K articles from 10 Sri Lankan news sites).
Hugging Face: [`sinhala-nlp/NSINA`](https://huggingface.co/datasets/sinhala-nlp/NSINA) (gated: accept the conditions first).
Licence: **CC-BY-SA-4.0**.

Citation: Hettiarachchi, H., Premasiri, D., Uyangodage, L., Ranasinghe, T. *NSINA: A News Corpus for Sinhala.* LREC-COLING 2024.

This project uses a seeded random 40 MB slice (`notebooks/01_download_data.ipynb`). All text is news, so the model will be biased toward news style and topics; this is discussed in the report.

## Run on Colab

Open `notebooks/00_colab_setup.ipynb` in Colab with a GPU runtime and run all cells.

## Run locally (CPU)

```bash
pip install -r requirements.txt
python -m pytest -q
```
