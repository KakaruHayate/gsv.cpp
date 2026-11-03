# Third-party notices

This project is MIT licensed (see [LICENSE](LICENSE)). It builds on, links against or
redistributes headers from the following projects:

| component | license | where |
|---|---|---|
| [ggml](https://github.com/ggml-org/ggml) / [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | linked at build time (`patches/` contains our ggml patch) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | MIT | `third_party/httplib/httplib.h` |
| [nlohmann/json](https://github.com/nlohmann/json) | MIT | `third_party/nlohmann/json.hpp`, `third_party/nlohmann/LICENSE.MIT` |
| [pocketfft](https://gitlab.mpcdf.mpg.de/mtr/pocketfft) | BSD-3-Clause | `third_party/pocketfft_hdronly.h` |
| [xxHash](https://github.com/Cyan4973/xxHash) | BSD-2-Clause | `third_party/xxhash/xxhash.h` |
| [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT | model weights derived from its checkpoints; see `licenses/GPT-SoVITS-LICENSE.txt` |

Model weights published at <https://huggingface.co/KakaruHayate/GSV.CPP> are conversions of
GPT-SoVITS pretrained/fine-tuned checkpoints and remain under the upstream project's license.

The text frontend under rontend/ is a vendored copy of GPT-SoVITS's GPT_SoVITS/text
(MIT). Its G2PW components (	ext/g2pw/, and the g2pW.onnx model distributed separately on
Hugging Face) come from the G2PW project and are Apache-2.0 licensed.
