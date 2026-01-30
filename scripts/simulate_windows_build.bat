@echo off
REM Simulates Windows CI CPU build - run on Windows to verify before pushing
REM Replicates release.yml build-windows-cpu: FFmpeg, pip install, build_dsu
REM Usage: run from project root

echo === Simulating Windows CI build (CPU) ===
echo.

echo Step 1: Ensure FFmpeg full-shared (torchcodec needs DLLs)
python scripts\ensure_ffmpeg_windows.py > ffmpeg_path.txt
if errorlevel 1 goto fail
set /p FFMPEG_DIR=<ffmpeg_path.txt
del ffmpeg_path.txt
if not defined FFMPEG_DIR goto fail
set "PATH=%FFMPEG_DIR%;%PATH%"
echo   FFmpeg: %FFMPEG_DIR%
echo.

echo Step 2: pip install -r requirements-cpu.txt
pip install --upgrade pip
pip install -r requirements-cpu.txt
if errorlevel 1 goto fail
echo.

echo Step 3: Test optional package imports (build_dsu does this)
python scripts\test_optional_imports.py
if errorlevel 1 goto fail
echo.

echo Step 4: Run build_dsu.py
python build_dsu.py
if errorlevel 1 goto fail
echo.

echo === BUILD OK ===
exit /b 0

:fail
echo.
echo === BUILD FAILED ===
exit /b 1
