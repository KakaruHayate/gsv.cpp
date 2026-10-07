@echo off
REM Build AR bench + engine parity test (Release, CPU+Vulkan) against llama.cpp/build-vk-rel
REM Output: tests/rel/bench_ar_rel.exe and tests/rel/test_ar_engine_rel.exe
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib (
  echo [error] missing llama.cpp/build-vk-rel
  exit /b 1
)
if not exist tests\rel mkdir tests\rel

set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

cl /nologo /O2 /EHsc /W1 %INC% tests\bench_ar.cpp src\gsv_ar.cpp src\gsv_sampler.cpp /Fe:tests\rel\bench_ar_rel.exe /Fo:tests\rel\ /link %LIBS% || exit /b 1
cl /nologo /O2 /EHsc /W1 %INC% tests\test_ar_engine.cpp src\gsv_ar.cpp src\gsv_sampler.cpp /Fe:tests\rel\test_ar_engine_rel.exe /Fo:tests\rel\ /link %LIBS% || exit /b 1
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\rel\ >nul
echo BUILD_OK -^> tests/rel/bench_ar_rel.exe tests/rel/test_ar_engine_rel.exe
exit /b 0

:novars
echo [error] vcvars64.bat not set; set VCVARS or edit this script
exit /b 1
