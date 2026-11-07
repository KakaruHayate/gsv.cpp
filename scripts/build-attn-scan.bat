@echo off
REM Build probe_attn_scan (single-shot DiT forward on a golden fixture, for attention stats).
setlocal
where cl >nul 2>&1
if not errorlevel 1 goto gsv_have_cl
if not defined VCVARS set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" goto gsv_no_vcvars
call "%VCVARS%" >nul 2>&1
goto gsv_have_cl
:gsv_no_vcvars
echo [error] vcvars64.bat not found, set VCVARS or edit this script
exit /b 1
:gsv_have_cl
cd /d "%~dp0.."

set REL=llama.cpp\build-vk-rel
if not exist "%REL%\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib" (
  echo [error] %REL% missing - build the Vulkan ggml first ^(see docs/build.md^)
  exit /b 1
)
set LIBS=%REL%\ggml\src\Release\ggml-base.lib %REL%\ggml\src\Release\ggml-cpu.lib %REL%\ggml\src\Release\ggml.lib %REL%\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set INC=/utf-8 /std:c++17 /O2 /arch:AVX2 /EHsc /W1 /openmp /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

if not exist tests\relcpu mkdir tests\relcpu
cl /nologo %INC% tests\probe_attn_scan.cpp src\gsv_dit.cpp src\gsv_sampler.cpp /Fe:tests\relcpu\probe_attn_scan.exe /Fo:tests\relcpu\ /link %LIBS% ws2_32.lib || exit /b 1
echo BUILD_OK -^> tests\relcpu\probe_attn_scan.exe
exit /b 0
