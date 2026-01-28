@echo off
setlocal
echo ============================================================
echo DSU Build Script
echo ============================================================

cd /d "%~dp0"

echo Cleaning previous build...

REM Kill any running DSU executables that may lock dist\dsu\lib\library.zip
taskkill /F /IM dsu-demucs.exe 2>nul
taskkill /F /IM dsu-bsroformer.exe 2>nul
taskkill /F /IM dsu-audio-separator.exe 2>nul

REM Build into a unique TEMP folder to avoid:
REM - file locks from previous runs
REM - "Access is denied" from protected folders / AV
set DSU_BUILD_EXE_DIR=%TEMP%\dsu_build_%RANDOM%

REM Best-effort cleanup with a retry (avoid parentheses blocks to prevent cmd parser issues)
if exist "%DSU_BUILD_EXE_DIR%" rmdir /s /q "%DSU_BUILD_EXE_DIR%"
if exist "%DSU_BUILD_EXE_DIR%" ping -n 3 127.0.0.1 >nul
if exist "%DSU_BUILD_EXE_DIR%" rmdir /s /q "%DSU_BUILD_EXE_DIR%"
if exist "%DSU_BUILD_EXE_DIR%" echo ERROR: Failed to clean build output directory: %DSU_BUILD_EXE_DIR% & exit /b 1

echo Building with setup.py...
build_env\Scripts\python.exe setup.py build_exe
if errorlevel 1 echo ERROR: Build failed. & exit /b 1

REM Validate expected outputs exist before moving
if not exist "%DSU_BUILD_EXE_DIR%\dsu-demucs.exe" echo ERROR: Missing dsu-demucs.exe in %DSU_BUILD_EXE_DIR% & exit /b 1
if not exist "%DSU_BUILD_EXE_DIR%\dsu-bsroformer.exe" echo ERROR: Missing dsu-bsroformer.exe in %DSU_BUILD_EXE_DIR% & exit /b 1
if not exist "%DSU_BUILD_EXE_DIR%\dsu-audio-separator.exe" echo ERROR: Missing dsu-audio-separator.exe in %DSU_BUILD_EXE_DIR% & exit /b 1
if not exist "%DSU_BUILD_EXE_DIR%\lib\library.zip" echo ERROR: Missing lib\library.zip in %DSU_BUILD_EXE_DIR% & exit /b 1

REM Replace dist\dsu with fresh build output (best-effort)
if exist dist\dsu rmdir /s /q dist\dsu
if exist dist\dsu ping -n 3 127.0.0.1 >nul
if exist dist\dsu rmdir /s /q dist\dsu
if exist dist\dsu echo WARNING: Could not remove dist\dsu (locked). Build output is in %DSU_BUILD_EXE_DIR% & goto :after_move
if not exist dist mkdir dist
move /Y "%DSU_BUILD_EXE_DIR%" dist\dsu >nul

:after_move
REM Smoke test: executables must start
dist\dsu\dsu-demucs.exe --help >nul
if errorlevel 1 echo ERROR: dsu-demucs.exe failed to start. & exit /b 1
dist\dsu\dsu-bsroformer.exe --help >nul
if errorlevel 1 echo ERROR: dsu-bsroformer.exe failed to start. & exit /b 1

echo ============================================================
echo Build complete! Test with:
echo   dist\dsu\dsu-demucs.exe --help
echo   dist\dsu\dsu-demucs.exe --worker
echo ============================================================
pause
