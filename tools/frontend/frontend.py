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
_REPO_ROOT = os.environ.get("GSV_REPO_ROOT", os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "repo")))
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
    if not os.path.isdir(_GSV_DIR):
        raise RuntimeError("找不到 repo (%s); 用 GSV_REPO_ROOT 指定" % _GENV_ROOT_HINT)
    # bert_path: 给 chinese2 的 G2PWPinyin 用 (它只要 tokenizer, 不加载模型权重)
    os.environ.setdefault("bert_path", _default_bert_dir())
    os.environ.setdefault("version", "v2")
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    if _GSV_DIR not in sys.path:
        sys.path.insert(0, _GSV_DIR)
    prev = os.getcwd()
    try:
        # chinese2.py 在 **import 时** 就构造 G2PWPinyin(模型路径相对 cwd), 所以必须在
        # cwd = repo 根 的前提下把语言模块全部 import 完; 之后 clean_text 用的是已导入的模块。
        os.chdir(_REPO_ROOT)
        import text
        from text import cleaned_text_to_sequence
        from text.cleaner import clean_text
        from text.symbols2 import symbols
        import text.chinese2    # G2PW 在这里构造 (需要 cwd/模型在位)
        for _m in ("english", "japanese", "korean", "cantonese"):
            try:
                __import__("text." + _m)
            except Exception as _e:          # 缺可选依赖的语言先跳过, 用到时再报
                print("[frontend] note: text.%s 未加载 (%s)" % (_m, _e), file=sys.stderr)
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
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "models", "chinese-roberta-wwm-ext-large")


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
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.bert_path)
        return self._tok

    def __call__(self, text, language=None):
        lang = language or self.language
        mods = _text_mods
        # auto: 用 LangSegmenter 切成 (lang, text) 片段, 逐段 G2P 后拼接
        if lang == "auto":
            return self._auto(text)
        phones, word2ph, norm_text = self._one(text, lang)
        phone_ids = mods["to_seq"](phones, os.environ.get("version", "v2"))
        ids = self._bert_ids_for(lang, norm_text, word2ph)
        return phone_ids, word2ph, ids

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
        for l, t in segs:
            lang = l if l == "en" else "zh"      # repo: 无法区分中日韩汉字, 以用户输入为准
            phones, word2ph, norm_text = self._one(t, lang)
            all_ids += _text_mods["to_seq"](phones, os.environ.get("version", "v2"))
            all_w2p += word2ph
            norm_parts.append(norm_text)
        norm_text = "".join(norm_parts)
        # 全 zh 才给真特征; 混合语言下 repo 会把"每段的特征沿 dim=1 拼接"(zh 段真、其余全零),
        # 而 server 的 bert 命令只接受一段连续的 word2ph —— 混合情况先退化为全零并说明。
        langs = {l for l, _ in segs}
        if langs and all(l == "zh" for l in langs):
            ids = self._bert_ids(norm_text)
            self._check(norm_text, all_w2p, ids)
            return all_ids, all_w2p, ids
        if langs - {"zh"} and "zh" in langs:
            print("[frontend] warning: auto 模式下中英混合; BERT 特征退化为全零 "
                  "(repo 会逐段拼接, 本版未实现) —— 音素侧不受影响", file=sys.stderr)
        return all_ids, all_w2p, []

    def _one(self, text, lang):
        mods = _text_mods
        lang = lang.replace("all_", "")
        phones, word2ph, norm_text = mods["clean_text"](text, lang, os.environ.get("version", "v2"))
        if word2ph is None:      # en 等无 word2ph 的语言: 每个音素算一个字
            word2ph = [1] * len(phones)
        return phones, word2ph, norm_text

    def _bert_ids(self, norm_text):
        """BERT tokenizer 输出 (含 CLS/SEP)。只分词, 不前向 —— 前向在 C++ 侧。"""
        enc = self._tokenizer()(norm_text, return_tensors=None)
        return list(enc["input_ids"])

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
