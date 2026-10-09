@echo off
REM Build DiT/CFM parity/bench test against llama.cpp/build-rel (Release CPU)
REM Output: tests/relcpu/test_dit.exe + Release DLLs
REM   GSV_DIT_DEVICE=vulkan 时改用 scripts\build-dit-vk.bat (build-vk-rel)
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

cl /nologo /O2 /arch:AVX2 /openmp /EHsc /W1 %INC% tests\test_dit.cpp src\gsv_dit.cpp src\gsv_sampler.cpp /Fe:tests\relcpu\test_dit.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
copy /y llama.cpp\build-rel\bin\Release\ggml*.dll tests\relcpu\ >nul
echo BUILD_OK -^> tests\relcpu\test_dit.exe
