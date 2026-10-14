@echo off
REM Build the gsv_ggml_cli server: ggml (build-vk-rel, the only build with the audio-patch
REM fused ops) + IPC transport + engine facade. ORT is loaded at runtime, so no import lib.
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu
set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set SRC=tools\gsv_ggml_cli.cpp src\gsv_engine.cpp src\gsv_ar.cpp src\gsv_bert.cpp src\gsv_cond.cpp src\gsv_dit.cpp src\gsv_encp.cpp src\gsv_hubert.cpp src\gsv_mel.cpp src\gsv_refcode.cpp src\gsv_refenc.cpp src\gsv_sampler.cpp src\gsv_vocoder.cpp src\gsv_wns1.cpp src\ipc\frame.cpp src\ipc\reader.cpp src\ipc\socket.cpp src\ipc\hash.cpp src\ipc\discovery.cpp src\ipc\session.cpp
set INC=/utf-8 /std:c++17 /O2 /arch:AVX2 /EHsc /W1 /openmp /I src /I third_party /I third_party\onnxruntime /I third_party\xxhash /I third_party\nlohmann /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
cl /nologo %INC% %SRC% /Fe:tests\relcpu\gsv_ggml_cli.exe /Fo:tests\relcpu\ /link %LIBS% ws2_32.lib || exit /b 1
echo BUILD_OK -^> tests\relcpu\gsv_ggml_cli.exe
exit /b 0
