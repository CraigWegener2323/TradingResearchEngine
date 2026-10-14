@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-dashboard\Scripts\python.exe" (
    echo Dashboard environment was not found.
    echo Run: py -3.14 -m venv .venv-dashboard
    echo Then: .venv-dashboard\Scripts\python.exe -m pip install -e ".[dashboard]"
    pause
    exit /b 1
)

".venv-dashboard\Scripts\python.exe" -m streamlit run "src\trading_research_engine\dashboard.py"
if errorlevel 1 (
    echo.
    echo The dashboard stopped with an error. Review the message above.
    pause
)
