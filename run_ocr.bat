@echo off
setlocal

REM Get the directory of the batch script
set SCRIPT_DIR=%~dp0

REM Check for Python
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in the system's PATH.
    echo Please install Python 3.12 and ensure it is added to your PATH.
    pause
    exit /b 1
)

REM Set up virtual environment
set VENV_DIR=%SCRIPT_DIR%venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment and install dependencies
echo Activating virtual environment and installing dependencies...
call "%VENV_DIR%\Scripts\activate.bat"

REM Install torch separately first due to build dependencies
echo Installing torch...
pip install torch==2.9.0
if %errorlevel% neq 0 (
    echo Error: Failed to install torch.
    pause
    exit /b 1
)

REM Install remaining required packages
echo Installing remaining required packages...
pip install -r "%SCRIPT_DIR%requirements.txt"
if %errorlevel% neq 0 (
    echo Error: Failed to install remaining required packages.
    pause
    exit /b 1
)

REM Run the OCR script
echo Starting OCR process...
python "%SCRIPT_DIR%run_dpsk_ocr.py" %*

echo.
echo OCR process finished.
pause
endlocal
