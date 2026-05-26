@echo off
REM First-time setup: install Python 3.10 if missing, create venv, install deps, download weights.
REM Run this ONCE after git clone. Then use start.bat / query.bat.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY_VER=3.10.11"
set "PY_URL=https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
set "PY_INSTALLER=%TEMP%\python-%PY_VER%-amd64.exe"
set "PY_DIR=%LOCALAPPDATA%\Programs\Python\Python310"

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
    echo       Found via "py -3.10".
    goto :have_python
)

if exist "%PY_DIR%\python.exe" (
    set "PY310=%PY_DIR%\python.exe"
    echo       Found at: !PY310!
    goto :have_python
)

echo       Not installed. Downloading official installer from python.org...
echo       URL: %PY_URL%
curl -L --fail -o "%PY_INSTALLER%" "%PY_URL%"
if errorlevel 1 (
    echo.
    echo ERROR: Could not download Python installer.
    echo Check your internet connection, or download manually from:
    echo   %PY_URL%
    pause
    exit /b 1
)

echo       Installing Python %PY_VER% silently (takes ~30 seconds)...
"%PY_INSTALLER%" /quiet ^
    InstallAllUsers=0 ^
    PrependPath=1 ^
    Include_launcher=1 ^
    Include_test=0 ^
    Include_doc=0 ^
    Include_dev=0 ^
    Shortcuts=0
if errorlevel 1 (
    echo.
    echo ERROR: Python install failed. Try running this file as Administrator,
    echo or install manually from: %PY_URL%
    pause
    exit /b 1
)
del /q "%PY_INSTALLER%" 2>nul

if not exist "%PY_DIR%\python.exe" (
    echo.
    echo Python install reported success but python.exe not found at:
    echo   %PY_DIR%\python.exe
    pause
    exit /b 1
)
set "PY310=%PY_DIR%\python.exe"
echo       Installed at: !PY310!

:have_python

REM ---------- 2. Create venv ----------
echo [2/5] Creating virtual environment .venv\ ...
if not exist .venv (
    "!PY310!" -m venv .venv
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
