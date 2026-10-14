@echo off
setlocal
if not defined VCVARS set VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist %VCVARS% (
  echo [error] vcvars64.bat not found, set VCVARS or edit this script
  exit /b 1
)
call %VCVARS% >nul 2>&1
cd /d "%~dp0.."
if not exist tests\relcpu mkdir tests\relcpu
set INC=/utf-8 /std:c++17 /O2 /EHsc /W1 /I src /I third_party /I third_party\xxhash
cl /nologo %INC% tests\ipc_probe.cpp src\ipc\frame.cpp src\ipc\reader.cpp src\ipc\socket.cpp src\ipc\hash.cpp src\ipc\discovery.cpp /Fe:tests\relcpu\ipc_probe.exe /Fo:tests\relcpu\ /link ws2_32.lib || exit /b 1
echo BUILD_OK -^> tests\relcpu\ipc_probe.exe
exit /b 0
