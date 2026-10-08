@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\GPT-SoVITS\gsv.cpp
set LIBS=llama.cpp\build-vk-rel\ggml\src\Release\ggml-base.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml-cpu.lib llama.cpp\build-vk-rel\ggml\src\Release\ggml.lib llama.cpp\build-vk-rel\ggml\src\ggml-vulkan\Release\ggml-vulkan.lib
set INC=/utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src /openmp
cl /nologo /O2 /arch:AVX2 /EHsc /W1 %INC% tests\test_hubert_ggml.cpp src\gsv_hubert.cpp /Fe:tests\rel\test_hubert_ggml.exe /Fo:tests\rel\ /link %LIBS%
echo RC=%ERRORLEVEL%
