@echo off
REM Build mel/ref-preprocess parity test (Release CPU) against llama.cpp/build-rel
REM Output: tests/relcpu/test_mel.exe
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu
set LIBS=llama.cpp\build-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-rel\ggml\src\Release\ggml.lib
set INC=/utf-8 /O2 /arch:AVX2 /openmp /EHsc /W1 /I src /I third_party /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
cl /nologo %INC% tests\test_mel.cpp src\gsv_mel.cpp /Fe:tests\relcpu\test_mel.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
echo BUILD_OK -^> tests\relcpu\test_mel.exe
exit /b 0
