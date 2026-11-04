"""文本 -> (phones, word2ph, bert_ids)。多语言前端, 套用 repo 的 text/ 那一层。

分工 (见 docs/e2e_server.md):
  Python 前端  : 文本规整 -> 多音字消歧 (G2PW) -> 音素化 -> phones/word2ph; BERT 分词 (只分词)
  C++ server   : BERT 前向 -> word2ph 逐字重复 -> AR -> 条件段 -> DiT -> vocoder -> wav

为什么不自己重写音素化: repo 的 text/ 是与 ckpt 一起训练/标定的, 重写必然引入口径差。
本模块只做**编排**与语言路由, 具体 G2P 全部转调 repo 的函数。

关于 g2pflow 的对比结论 (docs/e2e_server.md §5): 保留 repo 的 G2PW (神经网络) 而不是
cpp-pinyin 词典滑窗, 但吸取其"显式 language scope"的思路 —— 这里 language 是显式参数,
不做隐式猜测 (auto 模式才回落到 LangSegmenter)。
"""
import os
import sys

# repo 的 text/chinese2.py 在 import 时就用**相对 cwd** 的路径去开 G2PWModel,
# 所以必须先把 cwd 切到 repo 根, 并把 repo/GPT_SoVITS 放进 sys.path。
# Code root: contains GPT_SoVITS/text (vendored copy of the upstream frontend).
# Override with GSV_REPO_ROOT to point at a full GPT-SoVITS checkout for development.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.environ.get("GSV_REPO_ROOT", _HERE)
_GSV_DIR = os.path.join(_REPO_ROOT, "GPT_SoVITS")

_text_mods = {}

# NLTK 数据 (英文 G2P 的 g2p_en 需要 cmudict + averaged_perceptron_tagger_eng)。
# 本机 nltk_data 里只有旧的 tagger 名, 而 nltk 3.10 要 `_eng`; 网络又下不动,
# 所以仓库里放一份本地副本 (scripts/fetch-nltk-data.bat 生成), 有就优先用它。
_LOCAL_NLTK = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".nltk_data")
if os.path.isdir(_LOCAL_NLTK):
    os.environ["NLTK_DATA"] = _LOCAL_NLTK


def _ensure_repo():
    """把 cwd/sys.path/env 准备好并 import repo 的 text 层 (幂等)。"""
    if _text_mods:
        return _text_mods
    # The text layer imports chinese2 while cwd is temporarily switched to the frontend package
    # directory, so a relative data path would resolve against the wrong directory (G2PW would
    # then try to download its model and crash). Pin them to absolute paths before that.
    for _k in ("GSV_G2PW_MODEL", "GSV_BERT_DIR", "NLTK_DATA"):
        if os.environ.get(_k):
            os.environ[_k] = os.path.abspath(os.environ[_k])
    if not os.path.isdir(_GSV_DIR):
        raise RuntimeError("找不到 repo (%s); 用 GSV_REPO_ROOT 指定" % _GENV_ROOT_HINT)
    # bert_path: 给 chinese2 的 G2PWPinyin 用 (它只要 tokenizer, 不加载模型权重)
    os.environ.setdefault("bert_path", _default_bert_dir())
    # G2PW 模型目录：默认在 <repo>/models/frontend/g2pw，其次 <repo>/models/G2PWModel
    if not os.environ.get("GSV_G2PW_MODEL"):
        for cand in (os.path.join(os.path.dirname(_HERE), "models", "frontend", "g2pw"),
                     os.path.join(os.path.dirname(_HERE), "models", "G2PWModel")):
            if os.path.isdir(cand):
                os.environ["GSV_G2PW_MODEL"] = cand
                break
    os.environ.setdefault("version", "v2")
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)   # gsv_bert_tokenizer.py (no transformers/torch)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if _GSV_DIR not in sys.path:
        sys.path.insert(0, _GSV_DIR)
    prev = os.getcwd()
    try:
        # chinese2.py 在 **import 时** 就构造 G2PWPinyin(模型路径相对 cwd), 所以必须在
        # cwd = repo 根 的前提下把语言模块全部 import 完; 之后 clean_text 用的是已导入的模块。
        os.chdir(_REPO_ROOT)
        # Windows 上 g2pk2 依赖 eunjeon（老包，py3.10 无轮子）；repo 的 korean.py 在
        # eunjeon 缺失时会让 g2pk2.get_mecab() 执行 `raise print(...)` → TypeError。
        # 用 python-mecab-ko（`mecab` 模块）顶上，并跳过 eunjeon 检查。
        try:
            import g2pk2 as _g2pk2

            def _check_mecab(self):
                return None

            def _get_mecab(self):
                import mecab
                return mecab.MeCab()

            _g2pk2.G2p.check_mecab = _check_mecab
            _g2pk2.G2p.get_mecab = _get_mecab
        except Exception:
            pass
        import text
        from text import cleaned_text_to_sequence
        from text.cleaner import clean_text
        from text.symbols2 import symbols
        import text.chinese2    # G2PW 在这里构造 (需要 cwd/模型在位)
        # 其它语言**不要**在这里 import：repo 的 cleaner.clean_text 会在用到时
        # 自己 __import__("text."+module)；而 text.english 一 import 就是 ~7 s
        # (nltk/g2p_en)，服务化冷启动把这几秒省掉。
        _text_mods.update({
            "clean_text": clean_text,
            "to_seq": cleaned_text_to_sequence,
            "symbols": symbols,
            "text": text,
        })
    finally:
        os.chdir(prev)
    return _text_mods


_GENV_ROOT_HINT = _REPO_ROOT


def _default_bert_dir():
    if os.environ.get("GSV_BERT_DIR"):
        return os.environ["GSV_BERT_DIR"]
    data = os.environ.get("GSV_FRONTEND_DATA")
    if data:
        cand = os.path.join(data, "bert_tokenizer")
        if os.path.isdir(cand):
            return cand
    for cand in (os.path.join(os.path.dirname(_HERE), "models", "frontend", "bert_tokenizer"),
                 os.path.join(os.path.dirname(_HERE), "models", "chinese-roberta-wwm-ext-large")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(os.path.dirname(_HERE), "models", "chinese-roberta-wwm-ext-large")


class Frontend:
    """文本 -> (phones, word2ph, bert_ids)。

        fe = Frontend()                       # 默认 zh (G2PW 开)
        phones, word2ph, ids = fe("你好")
        # phones   [P] int32  音素 id (symbols2 下标) —— 给 AR 与 enc_p
        # word2ph   [N] int32  每字音素数 (Σ = P)    —— 给 server 做 BERT 逐字重复
        # bert_ids [L] int32  BERT tokenizer 输出 (含 CLS/SEP) —— 给 server 跑 gsv_bert

    language 显式指定 (zh / en / ja / ko / yue), "auto" 时用 repo 的 LangSegmenter 分段后
    逐段处理再拼接 —— 与 TextPreprocessor.get_phones_and_bert 同构。
    """

    def __init__(self, language="zh", bert_path=None, tokenizer=None):
        _ensure_repo()
        self.language = language
        self.bert_path = bert_path or _default_bert_dir()
        self._tok = tokenizer

    def _tokenizer(self):
        if self._tok is None:
            # self-contained WordPiece tokenizer: no transformers / torch dependency
            from gsv_bert_tokenizer import BertWordPieceTokenizer
            self._tok = BertWordPieceTokenizer(os.path.join(self.bert_path, "vocab.txt"))
        return self._tok

    def __call__(self, text, language=None):
        """-> (phone_ids, word2ph, bert_ids, bert_segs)

        bert_segs 只在 auto 混合语言时非空：每段 {"ids","w2p","start"}，
        server 把每段的 BERT 特征 scatter 到 phones 轴上（其余位置全零），
        与 repo「zh 段真特征 + 非 zh 全零」的逐段拼接一致。
        """
        lang = language or self.language
        mods = _text_mods
        if lang == "auto":
            return self._auto(text)
        phones, word2ph, norm_text = self._one(text, lang)
        phone_ids = mods["to_seq"](phones, os.environ.get("version", "v2"))
        ids = self._bert_ids_for(lang, norm_text, word2ph)
        return phone_ids, word2ph, ids, []

    def _bert_ids_for(self, lang, norm_text, word2ph):
        """zh 走真 BERT; 其它语言 repo 直接给全零 (get_bert_inf), 所以返回空表即可。

        repo TextPreprocessor.get_bert_inf:  language == "zh" 才算特征, 否则
        torch.zeros((1024, len(phones)))。所以 en/ja/ko/yue 路径**不需要** BERT tokenizer,
        也不需要 server 跑 BERT —— 少一次前向。
        """
        if lang.replace("all_", "") != "zh":
            return []
        ids = self._bert_ids(norm_text)
        self._check(norm_text, word2ph, ids)
        return ids

    def _auto(self, text):
        """分段处理再拼接 —— 与 repo get_phones_and_bert 的 auto 分支同构。"""
        from text.LangSegmenter import LangSegmenter
        segs = []
        for tmp in LangSegmenter.getTexts(text):
            l = tmp["lang"]
            if segs and ((l == "en") == (segs[-1][0] == "en")):
                segs[-1] = (segs[-1][0], segs[-1][1] + tmp["text"])
            else:
                segs.append((l, tmp["text"]))
        all_ids, all_w2p, norm_parts = [], [], []
        bert_segs = []
        for l, t in segs:
            lang = l if l == "en" else "zh"      # repo: 无法区分中日韩汉字, 以用户输入为准
            phones, word2ph, norm_text = self._one(t, lang)
            start = len(all_ids)
            all_ids += _text_mods["to_seq"](phones, os.environ.get("version", "v2"))
            all_w2p += word2ph
            norm_parts.append(norm_text)
            if lang == "zh":
                try:
                    ids = self._bert_ids(norm_text)
                    self._check(norm_text, word2ph, ids)
                    bert_segs.append({"ids": ids, "w2p": word2ph, "start": start})
                except ValueError as e:
                    print("[frontend] warning: BERT 段跳过 (%s)" % e, file=sys.stderr)
        return all_ids, all_w2p, [], bert_segs

    def _one(self, text, lang):
        mods = _text_mods
        lang = lang.replace("all_", "")
        phones, word2ph, norm_text = mods["clean_text"](text, lang, os.environ.get("version", "v2"))
        if word2ph is None:      # en 等无 word2ph 的语言: 每个音素算一个字
            word2ph = [1] * len(phones)
        return phones, word2ph, norm_text

    def _bert_ids(self, norm_text):
        """BERT tokenizer 输出 (含 CLS/SEP)。只分词, 不前向 —— 前向在 C++ 侧。"""
        tok = self._tokenizer()
        ids = tok.convert_tokens_to_ids(["[CLS]"] + tok.tokenize(norm_text) + ["[SEP]"])
        return list(ids)

    @staticmethod
    def _check(norm_text, word2ph, ids):
        """repo 的 get_bert_feature 隐含要求: 去掉 CLS/SEP 后 token 数 == 字符数 == len(word2ph)。
        中文成立; 混入未收录的 ASCII 时可能不成立, 早报错好过让 server 静默出错。
        """
        n_tok = len(ids) - 2
        if n_tok != len(word2ph):
            raise ValueError(
                "BERT token 数 (%d, 去 CLS/SEP 后) 与字数 (%d) 不一致; "
                "norm_text 里可能含未逐字对应的片段 (ASCII/数字), "
                "repo 的 get_bert_feature 同样会在这里 assert 失败" % (n_tok, len(word2ph)))
