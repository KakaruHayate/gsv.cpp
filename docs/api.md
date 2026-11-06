# HTTP API

`gsv_server` exposes an api_v2-compatible subset over HTTP. The machine-readable description
is served by the server itself: `GET /docs` or `GET /openapi.json` (OpenAPI 3).

## Basics

- Default address: `http://127.0.0.1:9880` (see `--host` / `--port`).
- Auth: when started with `--token <T>`, every request must send `Authorization: Bearer <T>`
  or `?token=<T>`. Binding to a non-loopback address requires `--token`.
- File access: `ref_audio_path` must live under one of the `--allow-path` directories, or under
  the upload directory created by `POST /reference`. Otherwise the server answers `403`.
- Errors are JSON: `{"code": "<error_code>", "message": "<human readable>"}` with HTTP status
  `400` (bad request), `401` (auth), `403` (path), `500` (engine), `501` (not implemented),
  `503` (busy / sidecar down), `504` (timeout).
- Responses carry `X-Request-Id` (supply your own via the header if you want it echoed).

## POST /tts

Synthesize one text.

`text_lang` / `prompt_lang` accept `zh`, `en`, `ja`, `yue`, `ko`, `auto`, `auto_yue` and the
`all_*` spellings (`all_zh`, `all_ja`, ...). When an explicit single-language mode is given but the
text contains a script that cleaner would drop (Chinese/Japanese/Korean text with `en`, Latin text
with `zh`/`ja`/`yue`/`ko`), the request is routed through the segmenter instead, so 中英混合 and
friends produce the right phonemes. If the result contains no phonemes at all (e.g. Korean text
with `zh`) the request fails with `text produced no phonemes (lang=...)` - pick the matching
language or `auto`. `ref_audio_path` may contain non-ASCII characters (Chinese/Japanese paths are
converted from UTF-8 explicitly).

```json
{
  "text": "你好，欢迎使用语音合成系统。",
  "text_lang": "zh",
  "ref_audio_path": "/path/to/ref.wav",
  "prompt_text": "reference transcript",
  "prompt_lang": "zh",
  "sample_steps": 4,
  "cfg_rate": 0.0,
  "seed": 20261014,
  "stream": false
}
```

| field | type | default | notes |
|---|---|---|---|
| `text` | string | — | required, <= 4000 bytes |
| `text_lang` | string | `zh` | `zh` / `en` / `ja` / `ko` / `yue` |
| `ref_audio_path` | string | — | required unless a default reference was set |
| `prompt_text` | string | — | required unless a default reference was set |
| `prompt_lang` | string | `zh` | language of `prompt_text` |
| `voice` | string | current | registered voice name |
| `sample_steps` | int | per voice | 1..1000 (v5turbo 4, v5dev 32) |
| `cfg_rate` | float | per voice | <= 20 |
| `seed` | int | -1 | -1 = random |
| `top_k`, `top_p`, `temperature`, `repetition_penalty` | | AR defaults (15/1/1/1.35) | |
| `speed_factor` | float | 1.0 | (0, 4] |
| `text_split_method` | string | `cut5` | `cut0`..`cut5` |
| `fragment_interval` | float | 0.3 | silence between segments, seconds (0..5) |
| `dbcache` | string | `auto` | cache-dit: `off` / `auto` / `f4` / `f8` / `f16` |
| `dit_window` | int | 128 | DiTFastAttn window in frames (reference segment stays a global prefix; 0 = off) |
| `dit_cfg_off_t` | float | 0.7 | DiT alg: skip the Uncond branch for `t >=` this value; skip ratio = `1 - dit_cfg_off_t` |
| `dit_uncond_reuse` | int | 0 | DiT alg: 0 off / 1 reuse neg on odd steps / 2 linear extrapolation |
| `timeout_ms` | int | 0 | override engine lock / scheduler / AR wait timeout in ms (<= 0 = server default, `--lock-timeout-ms`) |
| `media_type` | string | `wav` | only `wav` |
| `stream` | bool | `false` | see below |

`GET /tts` accepts the same fields as query parameters.

Non-stream response: `audio/wav`, 48 kHz mono 16-bit, whole-request peak-normalized to 0.708.

### Streaming (`stream: true`)

`POST /tts` with `stream: true` answers `Transfer-Encoding: chunked` with `audio/wav`:

- the 44-byte WAV header is sent immediately (RIFF/data sizes are `0xFFFFFFFF`, as usual for
  streaming WAV; players read until the connection closes);
- PCM is flushed after every text segment, plus `fragment_interval` of silence between segments;
- segments use a finer splitter than `cut5` (sentence-sized, >= 4 chars, <= 60 chars per chunk),
  so the first audio arrives earlier; `text_split_method=cut0` disables splitting;
- streamed audio is **not** peak-normalized (per-segment normalization would pump the volume);
- streamed requests bypass the AR batching scheduler, so total time can be higher than
  non-streaming for the same text; use streaming for latency, not throughput.

Measured on the dev box (CPU, 3 sentences): header in 13 ms, first PCM after the first segment,
total roughly the sum of the segments.

## GET /health, GET /metrics

`/health` → readiness: `{"ok":true,"engine":"ready","voice":"default","sidecar":true,"queue":0,...}`.

`/metrics` → counters: `tts_requests`, `errors`, `jobs`, `timeouts`, `busy`, `reloads`,
`ar_batches`, `ar_tasks`, `sched_batches`, `sched_texts`, `queue`, `waiters`, `last_total_ms`.

## Batch

- `POST /batch` with `{"texts":[...], ...same fields as /tts}` → `{"job_id":"<id>","total":N}`.
- `GET /jobs/<id>` → state (`queued` / `running` / `done` / `canceled` / `failed`).
- `GET /jobs/<id>/result/<i>` → WAV of item `i`.
- `DELETE /jobs/<id>` → cancel / drop.

## References

- `GET /change_refer?refer_wav_path=...&prompt_text=...&prompt_lang=zh` (POST also accepts a JSON
  body) stores a **default reference**. Later `/tts` calls may omit `ref_audio_path` /
  `prompt_text` / `prompt_lang`.
- `GET /set_refer_audio?...` is an alias of `/change_refer`.
- `POST /reference` with a raw WAV body uploads a reference to the temp directory and returns
  `{"path": "...", "bytes": N}`; that path is implicitly allowed for `ref_audio_path`.

## Misc

- `GET /models` → voices and the current one; `?rescan=1` re-scans `models/voices`.
- `GET /progress` → coarse progress of the current synthesis:
  `{"stage":1,"cur":0,"total":0}` where stage is `0` idle, `1` text frontend, `2` AR,
  `3` DiT/CFM, `4` vocoder, `5` done; `cur/total` counts AR tokens or DiT steps.
  The GAG GUI polls this for its progress bar.
- `GET /control` without `command` → `400 {"message":"command is required ..."}`. This is the
  api_v2-compatible liveness probe (the upstream GUI treats `400` as "API ready"); prefer
  `/health` for new clients.
- `GET /control?command=exit` → graceful shutdown (used by tests); `?command=restart` reloads
  the current voice.
- `GET /asr` → `501`, ASR is not part of this engine.
- `GET /docs`, `GET /openapi.json` → OpenAPI 3 document.
- `GET /tts_ws` → `501`: this build has no WebSocket support; use `POST /tts` with
  `stream: true` instead.
- `--parent-pid PID` (CLI) makes the server exit when that process disappears; the Windows
  one-click package uses it so closing the GUI frees the GPU.

## Examples

```bash
# one-shot
curl -s -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' \
  -d '{"text":"你好","text_lang":"zh","ref_audio_path":"/models/ref.wav",
       "prompt_text":"参考音频文本","sample_steps":4}' -o out.wav

# streaming to a player
curl -sN -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' \
  -d '{"text":"第一句。第二句。","ref_audio_path":"/models/ref.wav",
       "prompt_text":"参考音频文本","stream":true}' | ffplay -autoexit -

# default reference, then synthesize
curl -s "http://127.0.0.1:9880/change_refer?refer_wav_path=/models/ref.wav&prompt_text=%E5%8F%82%E8%80%83"
curl -s -X POST http://127.0.0.1:9880/tts -H 'Content-Type: application/json' -d '{"text":"你好"}'
```
