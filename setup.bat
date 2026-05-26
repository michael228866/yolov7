@echo off
REM First-time setup: install Python 3.10 if missing, create venv, install deps, download weights.
REM Run this ONCE after git clone. Then use start.bat / query.bat.

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo  Head Counter - First-time Setup
echo ============================================================
echo.

REM ---------- 1. Ensure Python 3.10 is available ----------
echo [1/5] Looking for Python 3.10...
set "PY310=py -3.10"
%PY310% --version >nul 2>&1
if %errorlevel% equ 0 (
    echo       Found.
    goto :have_python
)

echo       Not installed. Attempting winget install (silent)...
where winget >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: winget is not available on this PC.
    echo You can either:
    echo   a^) Install "App Installer" from the Microsoft Store, then re-run setup.bat
    echo   b^) Install Python 3.10 manually:
    echo      https://www.python.org/downloads/release/python-31011/
    echo      ^(check "Add python.exe to PATH" during install^)
    echo.
    pause
    exit /b 1
)

winget install --id Python.Python.3.10 -e ^
    --silent ^
    --accept-package-agreements ^
    --accept-source-agreements ^
    --scope user
if errorlevel 1 (
    echo.
    echo winget install failed. Install Python 3.10 manually:
    echo   https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

REM PATH won't refresh in this shell. Try py launcher first, fall back to explicit paths.
%PY310% --version >nul 2>&1
if %errorlevel% equ 0 goto :have_python

set "PY310=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if exist "!PY310!" goto :have_python
set "PY310=%ProgramFiles%\Python310\python.exe"
if exist "!PY310!" goto :have_python

echo.
echo Python 3.10 was installed but this shell hasn't picked it up.
echo Please CLOSE this window and DOUBLE-CLICK setup.bat AGAIN.
echo.
pause
exit /b 0

:have_python
echo       Using: !PY310!

REM ---------- 2. Create venv ----------
echo [2/5] Creating virtual environment .venv\ ...
if not exist .venv (
    !PY310! -m venv .venv
    if errorlevel 1 goto :fail
)

REM ---------- 3. Activate + upgrade pip ----------
echo [3/5] Activating venv and upgrading pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :fail

REM ---------- 4. Install dependencies ----------
echo [4/5] Installing PyTorch (CUDA 11.8) and other packages...
echo       Downloads ~2.5 GB the first time. Please wait.
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 ^
    --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto :fail
pip install -r requirements.txt
if errorlevel 1 goto :fail

REM ---------- 5. Download head-detection weights ----------
echo [5/5] Downloading head-detection weights...
if not exist yolov8_head_medium.pt (
    curl -L -o yolov8_head_medium.pt ^
        https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/medium.pt
)
if not exist yolov8_head_nano.pt (
    curl -L -o yolov8_head_nano.pt ^
        https://github.com/Abcfsa/YOLOv8_head_detector/raw/main/nano.pt
)

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Next steps:
echo    start.bat   - launch the counter
echo    query.bat   - view today's report
echo ============================================================
echo.
pause
exit /b 0

:fail
echo.
echo Setup FAILED. See errors above.
pause
exit /b 1
