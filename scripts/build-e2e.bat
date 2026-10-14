@echo off
REM Build the end-to-end demo: CPU Release (ggml CPU) + ORT vocoder (runtime-loaded)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu
REM build-vk-rel 是唯一含 audio-patch 融合算子 (ggml_layernorm_affine/ggml_add_act) 的构建
REM (llama.cpp\build-rel 是旧构建, 无这两个符号); ORT 侧运行时 LoadLibrary, 无需 .lib
set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set SRC=tests\demo_e2e.cpp src\gsv_ar.cpp src\gsv_bert.cpp src\gsv_cond.cpp src\gsv_dit.cpp src\gsv_encp.cpp src\gsv_hubert.cpp src\gsv_mel.cpp src\gsv_refcode.cpp src\gsv_refenc.cpp src\gsv_sampler.cpp src\gsv_vocoder.cpp src\gsv_wns1.cpp
set INC=/utf-8 /O2 /arch:AVX2 /openmp /EHsc /W1 /I src /I third_party /I third_party\onnxruntime /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
cl /nologo %INC% %SRC% /Fe:tests\relcpu\demo_e2e.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
echo BUILD_OK -^> tests\relcpu\demo_e2e.exe
exit /b 0
