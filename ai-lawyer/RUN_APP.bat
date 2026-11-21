@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=python"
if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
"%PY%" -m pip install -q -r requirements.txt
echo  Paste your Gemini API key and press Enter (not saved anywhere):
set /p GEMINI_API_KEY=
"%PY%" -m streamlit run app.py
pause
