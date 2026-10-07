@echo off
REM Build BERT parity/bench test (Release, CPU+Vulkan) against llama.cpp/build-vk-rel
REM Output: tests/rel/test_bert_rel.exe + Release DLLs in the same folder
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib (
  echo [error] missing llama.cpp/build-vk-rel
  exit /b 1
)
if not exist tests\rel mkdir tests\rel

set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

cl /nologo /O2 /EHsc /W1 %INC% tests\test_bert_ggml.cpp src\gsv_bert.cpp /Fe:tests\rel\test_bert_rel.exe /Fo:tests\rel\ /link %LIBS% || exit /b 1
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\rel\ >nul
echo BUILD_OK -^> tests\rel\test_bert_rel.exe
