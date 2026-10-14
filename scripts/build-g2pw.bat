@echo off
REM Build g2pw (ORT ONNX) parity test - CPU Release into tests/relcpu
REM ORT is loaded at runtime (LoadLibrary), so no .lib is needed.
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu
set INC=/utf-8 /O2 /arch:AVX2 /openmp /EHsc /W1 /I src /I third_party /I third_party\onnxruntime
cl /nologo %INC% tests\test_g2pw.cpp src\gsv_g2pw.cpp /Fe:tests\relcpu\test_g2pw.exe /Fo:tests\relcpu\ /link || exit /b 1
echo BUILD_OK -^> tests\relcpu\test_g2pw.exe
exit /b 0
