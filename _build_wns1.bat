call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d J:\GPT-SoVITS\gsv.cpp
cl /nologo /O2 /EHsc /utf-8 /I src /I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src tests\test_wns1.cpp /Fe:tests\test_wns1.exe /Fo:tests\ /link llama.cpp\build-cpu\ggml\src\Debug\ggml-base.lib llama.cpp\build-cpu\ggml\src\Debug\ggml-cpu.lib llama.cpp\build-cpu\ggml\src\Debug\ggml.lib
