@echo off
REM Build the host-reference sanity tests (ORCATERM): test_hubert (host CNN+transformer)
REM and test_wns1 (host WaveNet). Release CPU + OpenMP (they use #pragma omp), DLLs copied.
REM Output: tests/relcpu/test_hubert_host.exe / test_wns1_host.exe
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu

set LIBS=llama.cpp\build-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-rel\ggml\src\Release\ggml.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src /openmp /arch:AVX2

cl /nologo /O2 /EHsc /W1 %INC% tests\test_hubert.cpp /Fe:tests\relcpu\test_hubert_host.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
cl /nologo /O2 /EHsc /W1 %INC% tests\test_wns1.cpp /Fe:tests\relcpu\test_wns1_host.exe /Fo:tests\relcpu\ /link %LIBS% || exit /b 1
copy /y llama.cpp\build-rel\bin\Release\ggml*.dll tests\relcpu\ >nul
echo BUILD_OK -^> tests\relcpu\test_hubert_host.exe tests\relcpu\test_wns1_host.exe
