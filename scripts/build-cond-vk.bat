@echo off
REM Link the cond-segment tests against the Vulkan ggml build (Release)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist llama.cpp\build-vk-rel\CMakeCache.txt (
  echo [error] missing llama.cpp/build-vk-rel - configure it first, see docs/benchmark_ar.md section 5
  exit /b 1
)
set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
for %%T in (test_cond_rvq test_cond_bridge) do (
  cl /nologo /O2 /EHsc /W1 %INC% tests\%%T.cpp src\gsv_cond.cpp /Fe:tests\rel\%%T.exe /Fo:tests\rel\ /link %LIBS%
  if errorlevel 1 exit /b 1
)
echo VK_COND_TESTS_OK
