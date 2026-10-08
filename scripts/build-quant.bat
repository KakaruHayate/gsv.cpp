@echo off
REM Build quantize_gguf.exe (CPU ggml, Release)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib
set INC=/utf-8 /I llama.cpp\ggml\include /I llama.cpp\ggml\src
cl /nologo /O2 /EHsc /W1 %INC% tools\quantize_gguf.cpp /Fe:tools\quantize_gguf.exe /Fo:tools\ /link %LIBS%
if errorlevel 1 exit /b 1
echo QUANT_TOOL_OK
