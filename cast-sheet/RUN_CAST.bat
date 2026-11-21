@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
echo.
echo  Cast Sheet - resolving speaker labels with Gemini
echo  Paste your Gemini API key and press Enter (not saved anywhere):
set /p GEMINI_API_KEY=
"%PY%" -u run.py --input "data/I Married a Broken Soldier. He was a Duke"
echo.
echo  Finished. Results are in the outputs folder.
pause
