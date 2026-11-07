"""g2ps (numpy nano transformer) polyphone backend, drop-in for G2PWOnnxConverter.

Why: the ONNX G2PW path needs onnxruntime plus a 608 MB `g2pW.onnx` + dicts, and its
per-sentence cost dominates the text sidecar. `g2ps` bundles a 2.1 MB numpy-only model
(`nano`) with a word-level pinyin cache; the polyphone decisions for common Chinese
sentences are identical to G2PW, at a fraction of the cost.

Interface contract (must match what `text/g2pw/g2pw.py` expects from
`G2PWOnnxConverter`):

    converter = G2pSConverter(style="pinyin")
    rows = converter(["第一，我们开始"])       # -> [["di4", "yi1", ",", "wo3", ...]]

`rows[i]` has exactly `len(sentences[i])` entries so the caller can index it per
character; positions are pinyin in tone3 form, or the original character when the
position is not a Han char. A `None` entry means "no opinion, fall back to
pypinyin" (the same convention as the ONNX backend).

Selected with `GSV_G2P_BACKEND=g2ps` (default) or `GSV_G2P_BACKEND=g2pw` for the
original ONNX path; see `text/g2pw/g2pw.py`.
"""
from __future__ import annotations

import os

from pypinyin import Style, lazy_pinyin


class G2pSConverter:
    def __init__(self, model_dir: str = "nano", style: str = "pinyin", **_ignored):
        # `model_dir` keeps the same keyword as G2PWOnnxConverter; g2ps ships its model
        # inside the wheel, the argument only selects the bundled variant.
        from g2ps import G2pS

        self.engine = G2pS(model_dir if model_dir else "nano")
        self.polyphonic_chars = set(self.engine.char_cands.keys())
        self.style = style

    def _dict_row(self, sent: str):
        row = lazy_pinyin(sent, style=Style.TONE3, neutral_tone_with_five=True, errors="default")
        return row if len(row) == len(sent) else None

    def __call__(self, sentences, partial_results=None):
        if isinstance(sentences, str):
            sentences = [sentences]
        if partial_results is not None and len(partial_results) != len(sentences):
            raise ValueError("partial_results must match sentences length")

        out = []
        for i, sent in enumerate(sentences):
            preset = partial_results[i] if partial_results is not None else None
            base = self._dict_row(sent) or list(sent)
            row = list(base)

            # Phrase-level overrides (e.g. 第一 -> di4 yi1) arrive as a sparse list whose
            # holes are None. They must be merged ON TOP of the dictionary row, never used
            # as the row itself: returning the sparse list leaves None entries that
            # chinese2._g2p then subscripts (`pinyin[0]`), which crashes with
            # "TypeError: 'NoneType' object is not subscriptable" on inputs like 第一，.
            if preset is not None:
                for j, val in enumerate(preset):
                    if val is not None and j < len(row):
                        row[j] = val

            if any(ch in self.polyphonic_chars for ch in sent):
                try:
                    _, pairs = self.engine.convert(sent)
                except Exception:
                    pairs = []
                for j, (_ch, py) in enumerate(pairs):
                    if py is not None and j < len(row) and preset is None:
                        row[j] = py.replace("u:", "v")

            out.append(row)
        return out


def enabled() -> bool:
    return os.environ.get("GSV_G2P_BACKEND", "g2ps").strip().lower() != "g2pw"
