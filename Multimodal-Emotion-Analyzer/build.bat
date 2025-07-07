@echo off
REM Multimodal Emotion Analyzer Build Script for Windows
REM This script automates the build process on Windows

setlocal enabledelayedexpansion

echo Multimodal Emotion Analyzer Build Script for Windows
echo ===================================================

REM Check if CMake is installed
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake is not installed or not in PATH
    echo Please install CMake from https://cmake.org/download/
    exit /b 1
)

REM Check if Visual Studio or Build Tools are available
where cl >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Visual Studio compiler not found in PATH
    echo Trying to find Visual Studio installation...
    
    REM Try to find and setup Visual Studio environment
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo [ERROR] Visual Studio not found. Please install Visual Studio 2019 or newer
        exit /b 1
    )
)

REM Parse command line arguments
set BUILD_TYPE=Release
set GENERATOR=Visual Studio 17 2022
set ARCHITECTURE=x64
set CLEAN_BUILD=0

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--debug" (
    set BUILD_TYPE=Debug
) else if "%~1"=="--clean" (
    set CLEAN_BUILD=1
) else if "%~1"=="--vs2019" (
    set GENERATOR=Visual Studio 16 2019
) else if "%~1"=="--help" (
    echo Usage: %0 [options]
    echo Options:
    echo   --debug      Build in debug mode
    echo   --clean      Clean build directory before building
    echo   --vs2019     Use Visual Studio 2019 generator
    echo   --help       Show this help message
    exit /b 0
) else (
    echo [ERROR] Unknown option: %~1
    exit /b 1
)
shift
goto :parse_args
:args_done

echo [INFO] Build configuration:
echo   Generator: %GENERATOR%
echo   Architecture: %ARCHITECTURE%
echo   Build Type: %BUILD_TYPE%

REM Check for required model files
if not exist "models\haarcascade_frontalface_default.xml" (
    echo [WARNING] Haar cascade file not found
    echo [INFO] Please ensure haarcascade_frontalface_default.xml is in the models directory
    echo You can download it from: https://github.com/opencv/opencv/tree/master/data/haarcascades
)

REM Clean build directory if requested
if %CLEAN_BUILD%==1 (
    if exist "build" (
        echo [INFO] Cleaning build directory...
        rmdir /s /q build
    )
)

REM Create build directory
if not exist "build" (
    mkdir build
)

cd build

REM Configure with CMake
echo [INFO] Configuring with CMake...
cmake .. -G "%GENERATOR%" -A %ARCHITECTURE%
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    cd ..
    exit /b 1
)

REM Build the project
echo [INFO] Building project...
cmake --build . --config %BUILD_TYPE% --parallel
if errorlevel 1 (
    echo [ERROR] Build failed
    cd ..
    exit /b 1
)

cd ..

echo [INFO] Build completed successfully!
echo [INFO] Executable location: build\%BUILD_TYPE%\emotion_analyzer.exe

REM Check if executable was created
if exist "build\%BUILD_TYPE%\emotion_analyzer.exe" (
    echo [INFO] You can now run the application:
    echo   cd build\%BUILD_TYPE%
    echo   emotion_analyzer.exe
) else (
    echo [WARNING] Executable not found at expected location
    echo Please check the build output for errors
)

pause
