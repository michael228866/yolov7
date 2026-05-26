@echo off
REM Print today's IN/OUT report (or pass args like a date, --last 7, --sessions).
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python query_daily.py %*
pause
