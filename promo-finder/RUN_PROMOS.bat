@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
REM Model to use. Change this one line to swap models.
set PROMO_MODEL=gemini-3.8-flash
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
echo.
echo  Promo Finder - Gemini run on all episodes (model: %PROMO_MODEL%)
echo  ------------------------------------------
echo  Paste your Gemini API key and press Enter (it is not saved anywhere):
set /p GEMINI_API_KEY=
if not exist outputs mkdir outputs
"%PY%" -u run.py --input "data/I Married a Broken Soldier. He was a Duke" --resume
echo.
echo  (If it stopped early, just run this file again: finished episodes are skipped.)
echo  Finished. You can close this window.
pause
