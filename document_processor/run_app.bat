@echo off
cd /d "%~dp0"
"%USERPROFILE%\AppData\Roaming\Python\Python313\Scripts\streamlit.exe" run "app.py"
pause
