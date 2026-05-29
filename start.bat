@echo off
REM Launch the head counter. Activates the venv created by setup.bat.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Venv not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python count_inout_head.py --conf-thres 0.15 --auto-exposure 0.25 --exposure -4.5 --clahe --line-y 255 --source 1 --lite --api-url https://doorway-counter-api.rd-02f.workers.dev/ingest/live/event --zone hualien-d %*
pause
