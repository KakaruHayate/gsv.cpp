"""Minimal BERT WordPiece tokenizer used by the G2PW frontend.

Why this exists: upstream `text/g2pw/onnx_api.py` builds a HuggingFace
`AutoTokenizer`, which pulls in `transformers` (and, if it happens to be
installed, `torch`). The frontend only needs `tokenize()` and
`convert_tokens_to_ids()`, so a ~200-line WordPiece implementation is enough and
keeps the sidecar free of torch/transformers.

It mirrors `BertTokenizer(do_lower_case=True, tokenize_chinese_chars=True)` from
the `chinese-roberta-wwm-ext-large` / `bert-base-chinese` family; verified token
for token against HuggingFace on mixed Chinese/English text.
"""
import unicodedata


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in ("Cc", "Cf")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_chinese_char(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    )


class BertWordPieceTokenizer:
    def __init__(self, vocab_path: str, do_lower_case: bool = True, max_chars: int = 100):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                tok = line.rstrip("\n")
                if tok and tok not in self.vocab:
                    self.vocab[tok] = idx
        self.unk_token = "[UNK]"
        self.unk_id = self.vocab.get(self.unk_token, 100)
        self.do_lower_case = do_lower_case
        self.max_input_chars_per_word = max_chars

    # -- public API used by text/g2pw -------------------------------------
    def tokenize(self, text: str):
        tokens = []
        for word in self._basic_tokenize(text):
            tokens.extend(self._wordpiece(word))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_id)
        return [self.vocab.get(t, self.unk_id) for t in tokens]

    # -- internals ---------------------------------------------------------
    def _basic_tokenize(self, text: str):
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        out = []
        for token in text.strip().split():
            out.extend(self._run_split_on_punc(token))
        return [w for w in out if w]

    def _clean_text(self, text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            out.append(" " if _is_whitespace(ch) else ch)
        return "".join(out)

    def _tokenize_chinese_chars(self, text: str) -> str:
        out = []
        for ch in text:
            if _is_chinese_char(ord(ch)):
                out.append(" ")
                out.append(ch)
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    def _run_split_on_punc(self, text: str):
        out, cur = [], []
        for ch in text:
            if _is_punctuation(ch):
                if cur:
                    out.append("".join(cur))
                    cur = []
                out.append(ch)
            else:
                cur.append(ch)
        if cur:
            out.append("".join(cur))
        return out

    def _wordpiece(self, word: str):
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]
        if self.do_lower_case:
            word = self._lower(word)
        pieces, start = [], 0
        while start < len(word):
            end = len(word)
            found = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    found = sub
                    break
                end -= 1
            if found is None:
                return [self.unk_token]
            pieces.append(found)
            start = end
        return pieces

    @staticmethod
    def _lower(word: str) -> str:
        # bert's BasicTokenizer: lower + strip accents (NFD, drop Mn)
        word = unicodedata.normalize("NFD", word.lower())
        return "".join(c for c in word if unicodedata.category(c) != "Mn")
