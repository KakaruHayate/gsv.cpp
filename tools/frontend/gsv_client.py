# gsv_ggml_cli 的协议客户端。
#
# 传输设计照搬 KakaruHayate/openutau-vst-bridge (MPL-2.0) 的 PROTOCOL.md:
#   - server 在 %TEMP%/gsv-ggml/servers/*.json 写 discovery 文件 (port/name/apiVersion);
#   - 控制面 = LF 结尾的 UTF-8 行:  request:<uuid>:<kind> <json> / response:<uuid> <json> /
#     notification:<kind> <json> / close;
#   - 数据面 = 长度前缀二进制帧:  tensor <name> <hash> <len>\n<len bytes raw>;
#   - 大块 float32 (wav/mel/bert) 只走数据面, 绝不进 JSON (base64 有 +33% 开销且要整体拼内存)。
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid


class GsvError(RuntimeError):
    pass


class GsvClient:
    """一条到 gsv_ggml_cli 的连接。可用作上下文管理器。"""

    MAX_FRAME = 256 * 1024 * 1024          # 与 C++ 侧 kMaxFrameBytes 一致

    def __init__(self, host="127.0.0.1", port=None, autostart=False, exe=None,
                 models="models", device="", voc_device="", verbose=False, timeout=600.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.proc = None
        self.sock = None
        self._buf = b""

        if port is None:
            port = self._discover()
        if port is None:
            if not autostart:
                raise GsvError("没有发现 gsv_ggml_cli 服务 (discovery 目录为空); "
                               "用 autostart=True 或先手动启动")
            self.proc = self._spawn(exe or self._default_exe(), models, device, voc_device, verbose)
            port = self._wait_for_discovery(self.proc)
        self.port = port
        self.connect()

    # ---------------- discovery ----------------
    @staticmethod
    def discovery_dir():
        return os.path.join(tempfile.gettempdir(), "gsv-ggml", "servers")

    @staticmethod
    def _default_exe():
        # tools/frontend/gsv_client.py -> up 3 = 仓库根
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(here, "tests", "relcpu", "gsv_ggml_cli.exe")

    @classmethod
    def _discover(cls):
        """扫 discovery 目录, 返回第一个**真的在监听**的端口, 顺手删掉死掉的条目。

        只读文件是不够的: 崩掉的 server 会把文件留在那里, 客户端连上去只会被拒。
        探活成本很低 (一次 loopback connect), 比让调用方去排查"为什么连接被拒"划算。
        """
        d = cls.discovery_dir()
        if not os.path.isdir(d):
            return None
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    port = int(json.load(f)["port"])
            except Exception:
                continue
            if cls._probe(port):
                return port
            try:
                os.remove(path)          # stale: server 已经不在了
            except OSError:
                pass
        return None

    @staticmethod
    def _probe(port, timeout=0.5):
        s = socket.socket()
        s.settimeout(timeout)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False
        finally:
            s.close()

    def _spawn(self, exe, models, device, voc_device, verbose):
        args = [exe, "--models", models]
        if device:
            args += ["--device", device]
        if voc_device:
            args += ["--voc-device", voc_device]
        if verbose:
            args.append("--verbose")
        # 服务端日志必须排掉: subprocess.PIPE 不读的话管道缓冲 (~64 KB) 一满, server 就永久阻塞在
        # 写日志上 —— 表现为客户端在某个请求上超时。写到文件最省事, 也方便事后排查。
        log_path = os.environ.get("GSV_SERVER_LOG",
                                  os.path.join(tempfile.gettempdir(), "gsv_ggml_cli.log"))
        self.log_path = log_path
        self._log_fp = open(log_path, "wb")
        return subprocess.Popen(args, stdout=self._log_fp, stderr=subprocess.STDOUT)

    def _wait_for_discovery(self, proc, timeout=600.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                raise GsvError("server 提前退出 (code %s)" % proc.returncode)
            p = self._discover()
            if p:
                return p
            time.sleep(0.5)
        raise GsvError("等 discovery 文件超时")

    # ---------------- transport ----------------
    def connect(self):
        if self.sock is not None:
            return self
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        # server 一连上就发 notification:ready —— 等到它才算可用 (模型已载入)
        while True:
            line = self._read_line()
            if line is None:
                raise GsvError("连接在 ready 之前就断了")
            kind, name, payload = self._parse_control(line)
            if kind == "notification" and name == "ready":
                self.ready = json.loads(payload) if payload else {}
                return self
            if kind == "unknown":
                continue

    def close(self):
        try:
            if self.sock:
                self.sock.sendall(b"close\n")
                self.sock.close()
        except Exception:
            pass
        self.sock = None
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        try:
            if getattr(self, "_log_fp", None):
                self._log_fp.close()
                self._log_fp = None
        except Exception:
            pass

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _read_line(self):
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        if line.endswith(b"\r"):
            line = line[:-1]
        return line.decode("utf-8")

    def _read_exactly(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(self._buf)))
            if not chunk:
                raise GsvError("数据帧负载不足 (拿到 %d/%d)" % (len(self._buf), n))
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    @staticmethod
    def _parse_control(line):
        if line == "close":
            return "close", "", ""
        if line.startswith("response:"):
            rest = line[len("response:"):]
            uid, _, payload = rest.partition(" ")
            return "response", uid, payload
        if line.startswith("notification:"):
            rest = line[len("notification:"):]
            name, _, payload = rest.partition(" ")
            return "notification", name, payload
        if line.startswith("request:"):
            rest = line[len("request:"):]
            uid, _, tail = rest.partition(":")
            name, _, payload = tail.partition(" ")
            return "request", name, payload   # name 里返回 kind
        return "unknown", "", ""

    # ---------------- request ----------------
    def request(self, kind, params=None, tensors=None, want=None):
        """发一个 request, 收响应。

        tensors: {name: bytes} 先在数据面发出 (必须在控制行之前)。
        want:    期望回传的数据面帧名; None = 用响应里 `tensors` 字段声明的全部。
        返回 (data_dict, {name: bytes})。
        """
        if self.sock is None:
            raise GsvError("未连接 (先调 connect())")
        tensors = tensors or {}
        for name, raw in tensors.items():
            self.sock.sendall(("tensor %s %d %d\n" % (name, _xxh64(raw), len(raw))).encode("utf-8") + raw)
        uid = str(uuid.uuid4())
        body = json.dumps(params or {}, ensure_ascii=False)
        self.sock.sendall(("request:%s:%s %s\n" % (uid, kind, body)).encode("utf-8"))

        got = {}
        expected = None
        while True:
            line = self._read_line()
            if line is None:
                raise GsvError("连接断开 (等 %s 的响应)" % kind)
            if line.startswith("tensor "):
                parts = line.split(" ")
                if len(parts) != 4:
                    raise GsvError("畸形数据帧头: %r" % line)
                name, length = parts[1], int(parts[3])
                if length < 0 or length > self.MAX_FRAME:
                    raise GsvError("数据帧长度越界: %d" % length)
                got[name] = self._read_exactly(length)
                if expected is not None and all(w in got for w in expected):
                    return self._data, got
                continue
            k, a, payload = self._parse_control(line)
            if k == "notification":
                continue
            if k != "response" or a != uid:
                continue
            env = json.loads(payload) if payload else {}
            if not env.get("success", False):
                raise GsvError("%s 失败: %s" % (kind, env.get("error")))
            self._data = env.get("data", {})
            expected = want if want is not None else env.get("tensors", [])
            if not expected or all(w in got for w in expected):
                return self._data, got


def _xxh64(data):
    """XXH64(seed=0) —— 与 C++ 侧 gsvipc::Xxh64 一致。优先用 xxhash 包, 否则用纯 Python 实现。"""
    try:
        import xxhash
        return xxhash.xxh64(data).intdigest()
    except ImportError:
        return _xxh64_py(data)


# --- 纯 Python XXH64 (仅在前者不可用时) ---
_P1 = 11400714785074694791
_P2 = 14029467366897019727
_P3 = 1609587929392839161
_P4 = 9650029242287828579
_P5 = 2870177450012600261
_M64 = (1 << 64) - 1


def _rotl(x, r):
    return ((x << r) | (x >> (64 - r))) & _M64


def _round(acc, inp):
    acc = (acc + inp * _P2) & _M64
    acc = _rotl(acc, 31)
    return (acc * _P1) & _M64


def _merge(acc, val):
    val = _round(0, val)
    acc ^= val
    return (acc * _P1 + _P4) & _M64


def _xxh64_py(data, seed=0):
    n = len(data)
    i = 0
    if n >= 32:
        v1 = (seed + _P1 + _P2) & _M64
        v2 = (seed + _P2) & _M64
        v3 = seed & _M64
        v4 = (seed - _P1) & _M64
        while i + 32 <= n:
            v1 = _round(v1, struct.unpack_from("<Q", data, i)[0]); i += 8
            v2 = _round(v2, struct.unpack_from("<Q", data, i)[0]); i += 8
            v3 = _round(v3, struct.unpack_from("<Q", data, i)[0]); i += 8
            v4 = _round(v4, struct.unpack_from("<Q", data, i)[0]); i += 8
        h = (_rotl(v1, 1) + _rotl(v2, 7) + _rotl(v3, 12) + _rotl(v4, 18)) & _M64
        h = _merge(h, v1); h = _merge(h, v2); h = _merge(h, v3); h = _merge(h, v4)
    else:
        h = (seed + _P5) & _M64
    h = (h + n) & _M64
    while i + 8 <= n:
        k = _round(0, struct.unpack_from("<Q", data, i)[0])
        h ^= k
        h = (_rotl(h, 27) * _P1 + _P4) & _M64
        i += 8
    if i + 4 <= n:
        h ^= (struct.unpack_from("<I", data, i)[0] * _P1) & _M64
        h = (_rotl(h, 23) * _P2 + _P3) & _M64
        i += 4
    while i < n:
        h ^= (data[i] * _P5) & _M64
        h = (_rotl(h, 11) * _P1) & _M64
        i += 1
    h ^= h >> 33
    h = (h * _P2) & _M64
    h ^= h >> 29
    h = (h * _P3) & _M64
    h ^= h >> 32
    return h
