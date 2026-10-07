@echo off
REM Rebuild the ggml libraries (base + cpu + vulkan) in llama.cpp/build-vk-rel
REM Needed after any change under llama.cpp/ggml (ops, shaders) before relinking tests/rel
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
cmake --build llama.cpp\build-vk-rel --target ggml --config Release -j 16
if errorlevel 1 exit /b 1
echo VK_GGML_OK
