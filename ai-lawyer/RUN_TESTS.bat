@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
echo.
echo  Nyay Saathi - running all 10 test cases with Gemini
echo  Paste your Gemini API key and press Enter (not saved anywhere):
set /p GEMINI_API_KEY=
"%PY%" -u run_tests.py --resume
echo.
echo  (If it stopped early, run this file again: finished cases are skipped.)
pause
