@echo off
REM 把本仓库的 G2PW 模型链接到 repo 期望的位置 —— 该路径被 repo 自己的
REM GPT_SoVITS/text/.gitignore 忽略, 所以放这里不会污染它的工作区。
REM (用 PowerShell 建 junction: Git Bash 里 mklink 的参数会被路径转换搞坏)
set LINK=J:\GPT-SoVITS\repo\GPT_SoVITS\text\G2PWModel
set TARGET=J:\GPT-SoVITS\gsv.cpp\models\G2PWModel
if exist "%LINK%" (
  echo already linked: %LINK%
  exit /b 0
)
powershell -NoProfile -Command "New-Item -ItemType Junction -Path '%LINK%' -Target '%TARGET%' | Out-Null"
if errorlevel 1 (
  echo [warn] junction failed
  exit /b 1
)
echo linked: %LINK%
