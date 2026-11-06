@echo off
REM Build gsv_server: HTTP service layer (cpp-httplib) + gsv_engine, no IPC.
REM Text frontend is the Python sidecar (tools/text_sidecar.py), started separately.
setlocal
where cl >nul 2>&1
if not errorlevel 1 goto gsv_have_cl
if not defined VCVARS set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto gsv_no_vcvars
call "%VCVARS%" >nul 2>&1
goto gsv_have_cl
:gsv_no_vcvars
echo [error] vcvars64.bat not found, set VCVARS or edit this script
exit /b 1
:gsv_have_cl
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu

REM --- ggml first ---------------------------------------------------------
REM The engine uses custom ops that only exist in the patched llama.cpp tree
REM (ggml_conv_direct_1d_fused, ggml_add_act, ggml_im2col_fast_1d,
REM ggml_layernorm_affine). Shipping a stale ggml-vulkan.dll silently breaks the
REM vocoder output, so always bring ggml up to date with the current source.
if not exist llama.cpp\build-vk-rel\CMakeCache.txt (
  echo [error] llama.cpp\build-vk-rel missing - configure it first ^(see docs/build.md^)
  exit /b 1
)
cmake --build llama.cpp\build-vk-rel --config Release --target ggml-base ggml-cpu ggml ggml-vulkan || exit /b 1
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\relcpu\ >nul

set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set SRC=tools\gsv_server.cpp src\gsv_engine.cpp src\gsv_ar.cpp src\gsv_bert.cpp src\gsv_cond.cpp src\gsv_dit.cpp src\gsv_encp.cpp src\gsv_hubert.cpp src\gsv_mel.cpp src\gsv_refcode.cpp src\gsv_refenc.cpp src\gsv_sampler.cpp src\gsv_vocoder_ggml.cpp src\gsv_wns1.cpp
set INC=/utf-8 /std:c++17 /O2 /arch:AVX2 /EHsc /W1 /openmp /I src /I third_party /I third_party\httplib /I third_party\xxhash /I third_party\nlohmann /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
cl /nologo %INC% %SRC% /Fe:tests\relcpu\gsv_server.exe /Fo:tests\relcpu\ /link %LIBS% ws2_32.lib || exit /b 1

REM --- app-local runtime -------------------------------------------------
set VCRT_SRC=%GSV_VCRT_DIR%
if not defined VCRT_SRC if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC" set VCRT_SRC=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Redist\MSVC
if defined VCRT_SRC if exist "%VCRT_SRC%\vcruntime140.dll" (
  copy /y "%VCRT_SRC%\vcruntime140.dll" tests\relcpu\ >nul
  copy /y "%VCRT_SRC%\vcruntime140_1.dll" tests\relcpu\ >nul
  copy /y "%VCRT_SRC%\msvcp140.dll" tests\relcpu\ >nul
  echo COPIED VC runtime -^> tests\relcpu
) else (
  echo [note] set GSV_VCRT_DIR to a folder with a newer vcruntime140.dll/msvcp140.dll
)

echo BUILD_OK -^> tests\relcpu\gsv_server.exe
exit /b 0
