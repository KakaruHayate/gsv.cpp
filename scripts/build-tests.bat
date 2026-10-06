@echo off
REM 构建 ggml (CPU, Debug) 并编译 AR 对拍测试
REM 需要: VS2019 BuildTools (vcvars64.bat), cmake >= 3.20
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] 未找到 vcvars64.bat, 请修改脚本中的 VCVARS 路径
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist llama.cpp\ggml\CMakeLists.txt (
  echo [error] 缺少 llama.cpp (含补丁), 请先按 README "获取 ggml 基线" 操作
  exit /b 1
)

if not exist llama.cpp\build-cpu\CMakeCache.txt (
  cmake -S llama.cpp -B llama.cpp\build-cpu -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF || exit /b 1
)
cmake --build llama.cpp\build-cpu --target ggml -j %NUMBER_OF_PROCESSORS% || exit /b 1

set LIBS=llama.cpp\build-cpu\ggml\src\Debug\ggml-base.lib llama.cpp\build-cpu\ggml\src\Debug\ggml-cpu.lib llama.cpp\build-cpu\ggml\src\Debug\ggml.lib
set INC=/I llama.cpp\ggml\include /I llama.cpp\ggml\src /I llama.cpp\src

cl /nologo /O2 /EHsc /W1 %INC% tests\test_ar_step0.cpp  /Fe:tests\test_ar_step0.exe  /Fo:tests\test_ar_step0.obj  /link %LIBS% || exit /b 1
cl /nologo /O2 /EHsc /W1 %INC% tests\test_ar_decode.cpp /Fe:tests\test_ar_decode.exe /Fo:tests\test_ar_decode.obj /link %LIBS% || exit /b 1
cl /nologo /O2 /EHsc /W1 %INC% tests\test_min_ffn.cpp    /Fe:tests\test_min_ffn.exe    /Fo:tests\test_min_ffn.obj    /link %LIBS% || exit /b 1

copy /y llama.cpp\build-cpu\bin\Debug\ggml*.dll tests\ >nul
echo.
echo [ok] 构建完成: tests\test_ar_step0.exe tests\test_ar_decode.exe
echo      运行: tests\test_ar_step0.exe models\gsv-ar-f32.gguf tests\golden
