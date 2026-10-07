@echo off
REM Build ggml with the CUDA backend (Release, sm75) - Ninja generator.
REM Why Ninja: this machine only has VS2019 + CUDA 13.0's MSBuild integration registered
REM (CUDA 13 does not support VS2019). The VS generator therefore compiles .cu with nvcc 13.0.
REM Ninja lets CMake drive nvcc 12.6 directly (supported with cl 19.29).
REM Output: llama.cpp/build-cuda-ninja (libs in ggml/src/Release, DLLs in bin/Release)
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
if not defined NINJA set NINJA=ninja.exe
if not defined CUDA126 set CUDA126=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."

if not exist "%NINJA%" (
  echo [error] ninja not found at %NINJA% ^(set NINJA=...^)
  exit /b 1
)

if not exist llama.cpp\build-cuda-ninja\build.ninja (
  cmake -S llama.cpp -B llama.cpp\build-cuda-ninja -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_CUDA=ON -DGGML_NATIVE=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 ^
    -DCMAKE_CUDA_COMPILER="%CUDA126%/bin/nvcc.exe" ^
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF
  if errorlevel 1 exit /b 1
)
cmake --build llama.cpp\build-cuda-ninja --target ggml -j 16
if errorlevel 1 exit /b 1
echo BUILD_OK -^> llama.cpp/build-cuda-ninja
