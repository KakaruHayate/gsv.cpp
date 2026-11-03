"""gsv text sidecar: HTTP G2P micro-service (no torch).

The C++ service layer owns the engine and calls this process for the
GPT-SoVITS text frontend (normalisation, G2PW, pinyin, phones, BERT ids).

    python tools/text_sidecar.py --port 9882

Protocol:
    POST /g2p    {"text": "...", "lang": "zh"}
                 -> {"phones":[int], "word2ph":[int], "bert_ids":[int], "norm_text":"..."}
    GET  /health -> {"ok": true}

Run it from the gsv.cpp root (the frontend resolves repo/ and models/ relative to
the script).  Python deps: the repo's text/ requirements (no torch).
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

_T0 = time.perf_counter()
from frontend.frontend import Frontend  # noqa: E402
_T_IMPORT = time.perf_counter() - _T0

_lock = threading.Lock()
_frontends = {}
_ready = threading.Event()          # zh frontend 预热完成
_warm_ms = {"v": 0.0}


def get_frontend(lang):
    with _lock:
        fe = _frontends.get(lang)
        if fe is None:
            fe = Frontend(language=lang)
            _frontends[lang] = fe
        return fe


def warm():
    t0 = time.perf_counter()
    get_frontend("zh")
    _warm_ms["v"] = (time.perf_counter() - t0) * 1000
    _ready.set()
    print("[sidecar] zh frontend warm in %.1fs (import %.1fs)"
          % (_warm_ms["v"] / 1000.0, _T_IMPORT), flush=True)


def g2p(text, lang):
    fe = get_frontend(lang)
    with _lock:
        phones, word2ph, ids, bert_segs = fe(text, language=lang)
    return {
        "phones": [int(x) for x in phones],
        "word2ph": [int(x) for x in word2ph],
        "bert_ids": [int(x) for x in (ids or [])],
        # auto 混合语言：逐段 BERT，server 按 start 把特征 scatter 到 phones 轴
        "bert_segs": [
            {"ids": [int(x) for x in s["ids"]],
             "w2p": [int(x) for x in s["w2p"]],
             "start": int(s["start"])}
            for s in (bert_segs or [])
        ],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gsv-text-sidecar/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[sidecar] %s\n" % (fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json(200, {"ok": True, "ready": _ready.is_set(),
                             "warm_ms": round(_warm_ms["v"], 1)})
        else:
            self._json(404, {"message": "not found"})

    def do_POST(self):
        if not self.path.startswith("/g2p"):
            self._json(404, {"message": "not found"})
            return
        if not _ready.wait(timeout=180):
            self._json(503, {"message": "text frontend still warming"})
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            text = req.get("text", "")
            lang = req.get("lang", "zh")
            t0 = time.perf_counter()
            out = g2p(text, lang)
            out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            self._json(200, out)
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"message": "%s: %s" % (type(e).__name__, e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9882)
    args = ap.parse_args()

    # 先监听、再在后台预热：C++ 服务可以同时启动，/health 立即可用，
    # /g2p 会等 _ready（首次请求最多等预热完成）。
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[sidecar] listening on http://%s:%d (warming zh frontend in background)"
          % (args.host, args.port), flush=True)
    threading.Thread(target=warm, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
