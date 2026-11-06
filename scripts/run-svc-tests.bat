@echo off
REM Build and run the unit tests for tools/svc_core.h (no ggml/models needed).
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
if not exist tests\relcpu mkdir tests\relcpu
set INC=/utf-8 /std:c++17 /O2 /EHsc /W1 /I src /I third_party /I third_party\httplib /I third_party\xxhash /I third_party\nlohmann /I tools
cl /nologo %INC% tests\test_svc.cpp /Fe:tests\relcpu\test_svc.exe /Fo:tests\relcpu\ || exit /b 1
tests\relcpu\test_svc.exe || exit /b 1
echo TESTS_OK
exit /b 0
