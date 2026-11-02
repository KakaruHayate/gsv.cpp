@echo off
REM 准备英文 G2P 需要的 NLTK 数据到仓库本地 .nltk_data (已 gitignore)。
REM 本机 nltk_data 里只有旧的 averaged_perceptron_tagger 名, 而 nltk 3.10 要 `_eng`;
REM 网络下载被策略挡住, 所以从已有的那份复制出 `_eng` 目录。nltk 3.8.1 认旧名, 两者都能用。
set DEST=%~dp0..\.nltk_data
set SRC=%APPDATA%\nltk_data
if not exist "%SRC%\corpora\cmudict" (
  echo [error] 找不到 %SRC%\corpora\cmudict; 先跑 python -m nltk.downloader cmudict
  exit /b 1
)
if not exist "%DEST%\corpora" mkdir "%DEST%\corpora"
if not exist "%DEST%\taggers" mkdir "%DEST%\taggers"
xcopy /E /I /Y /Q "%SRC%\corpora\cmudict" "%DEST%\corpora\cmudict" >nul
if exist "%SRC%\taggers\averaged_perceptron_tagger_eng" (
  xcopy /E /I /Y /Q "%SRC%\taggers\averaged_perceptron_tagger_eng" "%DEST%\taggers\averaged_perceptron_tagger_eng" >nul
) else (
  xcopy /E /I /Y /Q "%SRC%\taggers\averaged_perceptron_tagger" "%DEST%\taggers\averaged_perceptron_tagger_eng" >nul
)
echo nltk data -> %DEST%
