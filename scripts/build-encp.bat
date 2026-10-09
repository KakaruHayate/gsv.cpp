@echo off
REM Build enc_p parity/bench test (Release CPU) against llama.cpp/build-rel
REM Output: tests/relcpu/test_encp.exe + Release DLLs in the same folder
REM   (build-rel has GGML_LLAMAFILE=ON + AVX2 - this is the correct build for CPU
REM    benchmarks; the ggml-cpu.dll in tests/ is a Debug build (/Od, 6-10x slower),
REM    do not use it for benchmarks)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist llama.cpp\build-rel\ggml\src\Release\ggml.lib (
  echo [error] missing llama.cpp/build-rel
  exit /b 1
)
if not exist tests\relcpu mkdir tests\relcpu

set LIBS=llama.cpp\build-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-rel\ggml\src\Release\ggml.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

cl /nologo /O2 /arch:AVX2 /openmp /EHsc /W1 %INC% tests\test_encp.cpp src\gsv_encp.cpp /Fe:tests\relcpu\test_encp.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
copy /y llama.cpp\build-rel\bin\Release\ggml*.dll tests\relcpu\ >nul
echo BUILD_OK -^> tests\relcpu\test_encp.exe
