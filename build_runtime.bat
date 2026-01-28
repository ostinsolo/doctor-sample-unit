@echo off
setlocal enabledelayedexpansion

REM =============================================================================
REM Doctor Sample Unit (DSU) - Shared Runtime Builder
REM Platform: Windows (CUDA or CPU)
REM =============================================================================
REM Creates a shared Python runtime with all dependencies for:
REM   - BS-RoFormer (28 models)
REM   - Audio-Separator (VR/MDX/MDXC models)
REM   - Demucs (8 models)
REM   - Apollo (audio restoration)
REM =============================================================================
REM Usage: build_runtime.bat [cuda|cpu]
REM   cuda - Build with CUDA 12.6 support (default)
REM   cpu  - Build CPU-only version
REM =============================================================================

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=cuda

echo =============================================================================
echo Doctor Sample Unit (DSU) - Shared Runtime Builder
echo =============================================================================
echo Platform: Windows %BUILD_TYPE%
echo Python: 3.10
echo =============================================================================
echo.

REM Check Python version
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Install Python 3.10 from https://python.org
    exit /b 1
)

REM Verify Python 3.10
python -c "import sys; exit(0 if sys.version_info[:2] == (3,10) else 1)" 2>nul
if errorlevel 1 (
    echo WARNING: Python 3.10 recommended for best compatibility
    echo Current version:
    python --version
    echo.
    echo Press Ctrl+C to cancel, or any key to continue anyway...
    pause >nul
)

REM Set paths
set SCRIPT_DIR=%~dp0
set RUNTIME_DIR=%SCRIPT_DIR%runtime

REM Check if runtime already exists
if exist "%RUNTIME_DIR%" (
    echo.
    echo WARNING: runtime directory already exists!
    echo Location: %RUNTIME_DIR%
    echo.
    set /p CONFIRM="Delete and rebuild? (y/N): "
    if /i not "!CONFIRM!"=="y" (
        echo Aborted.
        exit /b 1
    )
    echo Removing previous runtime...
    rmdir /s /q "%RUNTIME_DIR%"
)

REM Create virtual environment
echo.
echo [1/5] Creating Python virtual environment...
python -m venv "%RUNTIME_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

REM Activate
echo.
echo [2/5] Activating environment...
call "%RUNTIME_DIR%\Scripts\activate.bat"

REM Upgrade pip (suppress output)
echo.
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies based on build type
echo.
echo [4/5] Installing dependencies (%BUILD_TYPE%)...
echo This will take 10-15 minutes for CUDA, 5-10 minutes for CPU...
echo.

if "%BUILD_TYPE%"=="cuda" (
    pip install -r "%SCRIPT_DIR%requirements-cuda.txt"
) else (
    pip install -r "%SCRIPT_DIR%requirements-cpu.txt"
)

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies!
    exit /b 1
)

REM Verify installation
echo.
echo [5/5] Verifying installation...
echo.

"%RUNTIME_DIR%\Scripts\python.exe" -c "import torch; print(f'PyTorch: {torch.__version__}')"
"%RUNTIME_DIR%\Scripts\python.exe" -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if "%BUILD_TYPE%"=="cuda" (
    "%RUNTIME_DIR%\Scripts\python.exe" -c "import torch; print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else None"
)
"%RUNTIME_DIR%\Scripts\python.exe" -c "import numpy; print(f'NumPy: {numpy.__version__}')"
"%RUNTIME_DIR%\Scripts\python.exe" -c "import librosa; print(f'Librosa: {librosa.__version__}')"
"%RUNTIME_DIR%\Scripts\python.exe" -c "import soundfile; print(f'SoundFile: {soundfile.__version__}')"
"%RUNTIME_DIR%\Scripts\python.exe" -c "import demucs; print(f'Demucs: {demucs.__version__}')"
"%RUNTIME_DIR%\Scripts\python.exe" -c "import audio_separator; print(f'Audio-Separator: {audio_separator.__version__}')"

REM -----------------------------------------------------------------------------
REM Optional performance deps: SDPA + SageAttention (CUDA builds only)
REM Fail the build if these checks fail, so we don't ship a runtime that silently
REM falls back to slower paths.
REM -----------------------------------------------------------------------------
if "%BUILD_TYPE%"=="cuda" (
    echo.
    echo Verifying PyTorch SDPA on CUDA...
    "%RUNTIME_DIR%\Scripts\python.exe" "%SCRIPT_DIR%tests\test_sdpa_call.py"
    if errorlevel 1 (
        echo ERROR: PyTorch SDPA check failed in runtime environment.
        exit /b 1
    )

    echo.
    echo Verifying SageAttention import + kernel...
    "%RUNTIME_DIR%\Scripts\python.exe" "%SCRIPT_DIR%tests\check_sageattention.py"
    if errorlevel 1 (
        echo ERROR: SageAttention is missing or failed to import in runtime environment.
        exit /b 1
    )
    "%RUNTIME_DIR%\Scripts\python.exe" "%SCRIPT_DIR%tests\test_sageattention_kernel.py"
    if errorlevel 1 (
        echo ERROR: SageAttention kernel test failed in runtime environment.
        exit /b 1
    )
)

REM Deactivate
call "%RUNTIME_DIR%\Scripts\deactivate.bat" 2>nul

echo.
echo =============================================================================
echo BUILD COMPLETE!
echo =============================================================================
echo.
echo Runtime location: %RUNTIME_DIR%
echo Python executable: %RUNTIME_DIR%\Scripts\python.exe
echo.
echo Next steps:
echo   1. Build DSU executables:
echo      %RUNTIME_DIR%\Scripts\python.exe build_dsu_launchers.py
echo.
echo   2. Test a worker:
echo      %RUNTIME_DIR%\Scripts\python.exe workers\bsroformer_worker.py --help
echo.
echo =============================================================================

pause
