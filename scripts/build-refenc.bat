@echo off
REM Build ref_enc parity test (Release) - CPU (build-rel) into tests/relcpu,
REM and CPU+Vulkan (build-vk-rel) into tests/rel
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu

set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

set LIBSC=llama.cpp\build-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-rel\ggml\src\Release\ggml.lib
cl /nologo /O2 /arch:AVX2 /openmp /EHsc /W1 %INC% tests\test_refenc.cpp src\gsv_refenc.cpp /Fe:tests\relcpu\test_refenc.exe /Fo:tests\relcpu\ /link %LIBSC% || exit /b 1
copy /y llama.cpp\build-rel\bin\Release\ggml*.dll tests\relcpu\ >nul

set LIBSV=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
cl /nologo /O2 /EHsc /W1 %INC% tests\test_refenc.cpp src\gsv_refenc.cpp /Fe:tests\rel\test_refenc.exe /Fo:tests\rel\ /link %LIBSV% || exit /b 1
copy /y llama.cpp\build-vk-rel\bin\Release\ggml*.dll tests\rel\ >nul
echo BUILD_OK -^> tests\relcpu\test_refenc.exe tests\rel\test_refenc.exe
