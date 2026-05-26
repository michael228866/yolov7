@echo off
REM Launch the head counter. Activates the venv created by setup.bat.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python count_inout_head.py %*
pause
