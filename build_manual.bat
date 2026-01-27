@echo off
echo ============================================================
echo DSU Build Script
echo ============================================================

cd /d "%~dp0"

echo.
echo Cleaning previous build...
if exist dist rmdir /s /q dist

echo.
echo Building with setup.py...
build_env\Scripts\python.exe setup.py build_exe

echo.
echo ============================================================
echo Build complete! Test with:
echo   dist\dsu\dsu-demucs.exe --help
echo   dist\dsu\dsu-demucs.exe --worker
echo ============================================================
pause
