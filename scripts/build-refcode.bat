@echo off
REM Build ref-audio -> prompt-token parity test (Release, CPU+Vulkan) against llama.cpp/build-vk-rel
REM Output: tests/rel/test_refcode.exe   (GSV_*_DEVICE=vulkan 切 GPU; 该 vk-rel 是含融合算子的构建,
REM 而 llama.cpp/build-rel 是旧构建(无 ggml_layernorm_affine/add_act), 故统一用 vk-rel)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\rel mkdir tests\rel
set SRC=tests\test_refcode.cpp src\gsv_refcode.cpp src\gsv_mel.cpp src\gsv_hubert.cpp
set INC=/utf-8 /O2 /arch:AVX2 /openmp /EHsc /W1 /I src /I third_party /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src
set LIBV=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
cl /nologo %INC% %SRC% /Fe:tests\rel\test_refcode.exe /Fo:tests\rel\ /link %LIBV% || exit /b 1
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\rel\ >nul
echo BUILD_OK -^> tests/rel/test_refcode.exe
exit /b 0
