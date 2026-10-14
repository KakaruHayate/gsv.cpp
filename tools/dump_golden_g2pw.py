# G2PW golden dump —— 用 repo 的 text/g2pw 前端跑一遍, 落盘 ONNX 的输入/输出
#   golden (float32/int64 raw):
#     g2pw.{case}.input_ids / token_type_ids / attention_masks / phoneme_masks / char_ids / position_ids
#     g2pw.{case}.probs          [N, 1305]  (repo ONNX 的输出)
#     g2pw.{case}.pred           [N]        (argmax, int32 存成 float32 便于统一读)
#   case 名: 每行文本 + query_id 组成一个 batch (与 C++ 侧 test_g2pw.cpp 同参数)
#
# 依赖 repo 的 pypinyin/transformers/opencc -> 在独立 uv 环境里跑 (不动 diffsinger env)。
# 注意: repo 的 onnx_api.py 会去 modelscope 下载模型; 我们已把模型放到 models/G2PWModel,
#       用 G2PW_MODEL_DIR 指过去即可。
import argparse, io, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.environ.get("GSV_REPO", os.path.abspath(os.path.join(ROOT, os.pardir, "repo")))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "GPT_SoVITS"))
os.chdir(os.path.join(REPO, "GPT_SoVITS"))

import onnxruntime as ort                                     # noqa: E402
from text.g2pw.dataset import get_phoneme_labels, prepare_onnx_input   # noqa: E402
from text.g2pw.onnx_api import G2PWOnnxConverter, _find_first_existing_file  # noqa: E402
from text.g2pw.utils import tokenize_and_map                   # noqa: E402


def dump(outdir, name, arr):
    a = np.ascontiguousarray(arr)
    a.tofile(os.path.join(outdir, name + ".bin"))


def load_audio(path):  # noqa: D103 - placeholder kept for parity with other dumps
    raise NotImplementedError


def build_ref(outdir, model_dir, cases, tokenizer):
    """跑 repo 的 ONNX, 落盘输入/输出"""
    os.makedirs(outdir, exist_ok=True)
    labels, char2phonemes = get_phoneme_labels(
        polyphonic_chars=[line.split("\t") for line in
                          open(os.path.join(model_dir, "POLYPHONIC_CHARS.txt"), encoding="utf-8")
                          .read().strip().split("\n")])
    chars = sorted(list(char2phonemes.keys()))
    char2id = {c: i for i, c in enumerate(chars)}
    char_phoneme_masks = {c: [1 if i in char2phonemes[c] else 0 for i in range(len(labels))]
                          for c in char2phonemes}

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(model_dir, "g2pW.onnx"), so, providers=["CPUExecutionProvider"])

    # 注意: dataset.prepare_onnx_input 的 dict key 是 "attention_masks"/"phoneme_masks",
    # 而 ONNX 图的输入名是 "attention_mask"/"phoneme_mask" —— repo 的 predict() 手工映射过。
    FEED = {
        "input_ids": "input_ids",
        "token_type_ids": "token_type_ids",
        "attention_mask": "attention_masks",
        "phoneme_mask": "phoneme_masks",
        "char_ids": "char_ids",
        "position_ids": "position_ids",
    }

    for name, texts, query_ids in cases:
        mi = prepare_onnx_input(tokenizer=tokenizer, labels=labels, char2phonemes=char2phonemes,
                                chars=chars, texts=texts, query_ids=query_ids,
                                use_mask=True, window_size=None, char2id=char2id,
                                char_phoneme_masks=char_phoneme_masks)
        if not mi:
            print("[%s] prepare_onnx_input 返回空" % name)
            continue
        feed = {graph_name: mi[dict_key] for graph_name, dict_key in FEED.items()}
        probs = sess.run([], feed)[0]
        preds = np.argmax(probs, axis=1)
        dump(outdir, "g2pw.%s.input_ids" % name, mi["input_ids"])
        dump(outdir, "g2pw.%s.token_type_ids" % name, mi["token_type_ids"])
        dump(outdir, "g2pw.%s.attention_masks" % name, mi["attention_masks"])
        dump(outdir, "g2pw.%s.phoneme_masks" % name, mi["phoneme_masks"])
        dump(outdir, "g2pw.%s.char_ids" % name, mi["char_ids"])
        dump(outdir, "g2pw.%s.position_ids" % name, mi["position_ids"])
        dump(outdir, "g2pw.%s.probs" % name, probs)
        dump(outdir, "g2pw.%s.pred" % name, preds.astype(np.float32))
        print("[%s] N=%d L=%d -> pred %s" % (name, len(texts), mi["input_ids"].shape[1], preds.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "models", "G2PWModel"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--bert-source", default=os.path.join(ROOT, "models", "chinese-roberta-wwm-ext-large"))
    args = ap.parse_args()

    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.bert_source)

    # 每个 case = (名字, texts, query_ids)。**一条 text 对应一个 query_id** (repo 的
    # prepare_onnx_input 就是这么约定的); 同一句要查多个多音字就把句子重复几遍。
    # query_id 指向文本中的**多音字**下标 (POLYPHONIC_CHARS.txt 里的字; 非多音字在 repo 侧走
    # monophonic/pypinyin 兜底, 不过 ONNX, 所以对拍用例必须落在多音字上)。
    # 下标 = code point 索引 (与 repo 的 str 索引一致), 已用 POLYPHONIC_CHARS.txt 逐个核对:
    cases = [
        # "行" (idx 4, xing2/hang2) —— 旅行
        ("single", ["我喜欢旅行"], [4]),
        # 三条一批, 变长 -> 右侧 pad; 分别指向 重/的/得
        ("batch", ["我喜欢旅行", "重庆的路很长", "得到什么东西"], [4, 0, 0]),
        # 同字不同上下文: "行" 在 idx 1 (银行 -> hang2) 与 idx 5 (行动 -> xing2)
        ("ctx", ["银行里可以行动吗", "银行里可以行动吗"], [1, 5]),
        # 含数字混排 (wordize_and_map 的 ASCII 分支); 了(idx2)/会(idx8)
        ("mixed", ["我吃了3个苹果会不会", "我吃了3个苹果会不会"], [2, 8]),
        # 多音字靠近句首 (position_id 小); 长(idx2) -> chang2 (长江)
        ("head", ["长江的水很清"], [2]),
        # 长句, 多音字分布在不同位置 (含标点); 的(idx2)/真(idx4)/风(idx6)
        ("long", ["这里的风景真的很漂亮，我们走了很久很久，大家都还不想回家呢",
                  "这里的风景真的很漂亮，我们走了很久很久，大家都还不想回家呢",
                  "这里的风景真的很漂亮，我们走了很久很久，大家都还不想回家呢"], [2, 4, 6]),
        # 同一句子同一 query 重复两遍 (repo 的 sentence_dedup 会合并成一次前向)
        ("dup", ["我喜欢旅行", "我喜欢旅行"], [4, 4]),
    ]
    build_ref(args.out, args.model_dir, cases, tok)
    print("g2pw golden done ->", args.out)


if __name__ == "__main__":
    main()
