# Text frontend (G2P)

This directory vendors the GPT-SoVITS text frontend so the sidecar no longer has to read from
a full GPT-SoVITS checkout.

```
frontend/
  GPT_SoVITS/text/          # verbatim copy of upstream GPT_SoVITS/text (MIT, RVC-Boss)
    g2pw/  zh_normalization/  en_normalization/  LangSegmenter/  ja_userdic/  *.py
  frontend.py               # our adapter: language routing + (phones, word2ph, bert_ids)
  gsv_bert_tokenizer.py     # self-contained BERT WordPiece tokenizer (no transformers/torch)
  requirements-zh.txt       # zh frontend deps
  requirements-multi.txt    # extra deps for en / ja / ko / yue
  sync_frontend.py          # refresh the vendored copy from a GPT-SoVITS checkout
```

## Torch / transformers

The frontend does **not** need torch, and after the tokenizer replacement it does not need
`transformers` either:

- upstream `text/g2pw/onnx_api.py` used `AutoTokenizer`, which pulls `transformers` (and torch
  when it happens to be installed, which is what broke the Linux sidecar);
- `gsv_bert_tokenizer.py` reproduces `BertTokenizer(do_lower_case=True, tokenize_chinese_chars=True)`
  — the variant used by `chinese-roberta-wwm-ext-large` / `bert-base-chinese` — with a plain
  `vocab.txt`.

Verified: with `transformers` removed from the venv, the sidecar produces the same
`phones` / `bert_ids` and the **same WAV SHA-256** as the old build (`395b0ab4…`, 3.40 s).

## Polyphone backend: g2ps (default) vs g2pw

| `GSV_G2P_BACKEND` | engine | needs | cost (4 vCPU, zh) |
|---|---|---|---|
| `g2ps` (default) | numpy nano transformer, 2.1 MB model shipped in the wheel | `g2ps`, `pypinyin-g2pS` | ~2.9 ms / sentence, 0.28 ms / char |
| `g2pw` | ONNX G2PW (`g2pW.onnx` + dicts) | `onnxruntime` + 608 MB of data | ~40-60 ms of the 101 ms frontend column in `docs/benchmark.md` |

Both are char-aligned (`row[i]` belongs to `sentences[i]`), tone3 pinyin, `None` = "fall back to
pypinyin". `g2ps` also merges phrase-level overrides into the pypinyin dictionary row instead of
using the sparse override list as the row, which previously crashed `chinese2._g2p` on inputs like
`第一，` (`TypeError: 'NoneType' object is not subscriptable`).

Verify a backend and time it:

```
python frontend/check_g2ps.py --backend g2ps --repeat 10    # prints per-case phones + ms/char
python frontend/check_g2ps.py --backend g2pw --repeat 10    # diff the two JSON dumps for parity
```

## Data files (not in git)

| what | size | where |
|---|---|---|
| `G2PWModel/` (`g2pW.onnx` + dicts) | 608 MB | HF `KakaruHayate/GSV.CPP`, `frontend/g2pw/` |
| BERT tokenizer (`vocab.txt`, `config.json`, `tokenizer_config.json`) | ~0.1 MB | HF `frontend/bert_tokenizer/` |
| `nltk_data` (cmudict + tagger, English only) | 9 MB | HF `frontend/nltk_data.tar.gz` |
| `ja_userdic/userdict.csv` | 36.8 MB | in git (upstream path, needed by `japanese.py`) |

Environment variables:

| variable | meaning |
|---|---|
| `GSV_G2PW_MODEL` | directory containing `g2pW.onnx`; default search: `<repo>/models/frontend/g2pw`, `<repo>/models/G2PWModel`, then `GPT_SoVITS/text/G2PWModel` |
| `GSV_BERT_DIR` | directory containing `vocab.txt`; default search: `<repo>/models/frontend/bert_tokenizer`, `<repo>/models/chinese-roberta-wwm-ext-large` |
| `GSV_FRONTEND_DATA` | optional root; if it has `bert_tokenizer/`, that is used |
| `NLTK_DATA` | nltk data directory (`nltk_data`), needed for English |
| `GSV_REPO_ROOT` | dev only: point at a full GPT-SoVITS checkout instead of this directory |

## Following upstream

```bash
python frontend/sync_frontend.py --repo /path/to/GPT-SoVITS          # copy + re-apply patches
python frontend/sync_frontend.py --repo /path/to/GPT-SoVITS --check  # diff only
```

The script copies `GPT_SoVITS/text` (skipping `G2PWModel/`, `__pycache__/`, `*.pyc`) and applies
two small edits:

1. `text/g2pw/onnx_api.py`: use `gsv_bert_tokenizer.BertWordPieceTokenizer` when importable,
   fall back to `AutoTokenizer` otherwise.
2. `text/chinese2.py`: `model_dir` honours `GSV_G2PW_MODEL`.

Review `git diff frontend/GPT_SoVITS/text` after syncing, then run the sidecar probe
(`POST /g2p`) for zh/en and compare a synth WAV hash; the upstream text code changes rarely,
but it is cheap insurance for phoneme tables.
