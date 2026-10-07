@echo off
REM Build ggml (CPU, Debug) and compile all parity tests
REM Requires: VS2019 BuildTools (vcvars64.bat), cmake >= 3.20
REM NOTE: /utf-8 is required (sources contain Chinese comments)
REM NOTE: this file must stay ASCII-only (cmd + UTF-8 comments break parsing)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, edit VCVARS in this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist llama.cpp\ggml\CMakeLists.txt (
  echo [error] llama.cpp with patch missing
  exit /b 1
)

if not exist llama.cpp\build-cpu\CMakeCache.txt (
  cmake -S llama.cpp -B llama.cpp\build-cpu -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
)
cmake --build llama.cpp\build-cpu --target ggml -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1

set LIBS=llama.cpp\build-cpu\ggml\src\Debug\ggml-base.lib llama.cpp\build-cpu\ggml\src\Debug\ggml-cpu.lib llama.cpp\build-cpu\ggml\src\Debug\ggml.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

for %%T in (test_ar_step0 test_ar_decode test_ar_batch test_ar_sampler test_min_ffn test_ar_engine test_fused_ops) do (
  cl /nologo /O2 /EHsc /W1 %INC% tests\%%T.cpp src\gsv_sampler.cpp src\gsv_ar.cpp /Fe:tests\%%T.exe /Fo:tests\ /link %LIBS%
  if errorlevel 1 exit /b 1
)
cl /nologo /O2 /EHsc /W1 %INC% tests\test_bert_ggml.cpp src\gsv_bert.cpp /Fe:tests\test_bert_ggml.exe /Fo:tests\ /link %LIBS%
if errorlevel 1 exit /b 1
cl /nologo /O2 /EHsc /W1 %INC% tests\test_cond_rvq.cpp src\gsv_cond.cpp /Fe:tests\test_cond_rvq.exe /Fo:tests\ /link %LIBS%
if errorlevel 1 exit /b 1

copy /y llama.cpp\build-cpu\bin\Debug\ggml*.dll tests\ >nul
echo.
echo [ok] build done. parity:
echo      tests\test_ar_step0.exe   models\gsv-ar-f32.gguf tests\golden
echo      tests\test_ar_decode.exe  models\gsv-ar-f32.gguf tests\golden
echo      tests\test_ar_batch.exe   models\gsv-ar-f32.gguf tests\golden
echo      tests\test_ar_sampler.exe tests\golden
echo      tests\test_ar_engine.exe  models\gsv-ar-f32.gguf tests\golden
echo      tests\test_bert_ggml.exe  (GSV_BERT_DEVICE=vulkan for GPU)
