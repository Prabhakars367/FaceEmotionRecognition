# Windows Installation Guide - Multimodal Emotion Analyzer

This guide will help you complete the setup for your Multimodal Emotion Analyzer project. Based on your current setup, here's what you need to do.

## Current Status ✅

You already have:
- ✅ **OpenCV**: Installed at `C:\opencv\build\x64\vc16`
- ✅ **PortAudio**: Downloaded at `C:\Users\KIIT\Downloads\pa_stable_v190700_20210406 (1)\portaudio`
- ✅ **CMake**: Available in your system

## Still Need to Complete

### 1. Build PortAudio Library

Your PortAudio is downloaded but needs to be compiled:

**Option A: Using Visual Studio (Recommended)**
1. Navigate to your PortAudio folder:
   ```
   C:\Users\KIIT\Downloads\pa_stable_v190700_20210406 (1)\portaudio
   ```

2. Look for Visual Studio solution files in one of these locations:
   - `build\msvc\` folder
   - `msvc\` folder  
   - Root directory (`.sln` files)

3. Open the `.sln` file with Visual Studio 2022
4. Select **Release** configuration and **x64** platform
5. Build the solution (Build → Build Solution)

**Option B: Using CMake**
1. Open Command Prompt as Administrator
2. Navigate to PortAudio folder:
   ```cmd
   cd "C:\Users\KIIT\Downloads\pa_stable_v190700_20210406 (1)\portaudio"
   ```
3. Create build directory:
   ```cmd
   mkdir build
   cd build
   ```
4. Configure and build:
   ```cmd
   cmake -G "Visual Studio 17 2022" -A x64 ..
   cmake --build . --config Release
   ```

### 2. Verify Visual Studio 2022 Installation

Check if you have Visual Studio 2022 with C++ support:

1. **Check if installed**:
   - Look for "Visual Studio 2022" in Start Menu
   - Or check: `C:\Program Files\Microsoft Visual Studio\2022\`

2. **If not installed**, download from:
   - https://visualstudio.microsoft.com/downloads/
   - Select "Community" (free version)
   - During installation, choose "Desktop development with C++" workload

### 3. Set Up Developer Environment

**Option A: Use Developer Command Prompt**
1. Search for "Developer Command Prompt for VS 2022" in Start Menu
2. Run as Administrator

**Option B: Use Regular Command Prompt**
1. Open Command Prompt as Administrator
2. Run this command first:
   ```cmd
   "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
   ```

## Building Your Project

Once PortAudio is built and Visual Studio is ready:

### Step 1: Navigate to Project
```cmd
cd "C:\Users\KIIT\Downloads\emotion_analyzer_cpp\New folder"
```

### Step 2: Clean Previous Build (if any)
```cmd
rmdir /s /q build
```

### Step 3: Create Build Directory
```cmd
mkdir build
cd build
```

### Step 4: Configure with CMake
```cmd
cmake -G "Visual Studio 17 2022" -A x64 ..
```

### Step 5: Build the Project
```cmd
cmake --build . --config Release
```

## Expected Output

After successful build, you should see:
- `emotion_analyzer.exe` in `build\Release\` folder
- No compilation errors
- PortAudio and OpenCV libraries linked successfully

## Troubleshooting

### If PortAudio is not found:
1. Check that PortAudio was built successfully
2. Look for `portaudio_x64.lib` or similar files in PortAudio build output
3. Update the CMakeLists.txt paths if needed

### If OpenCV is not found:
1. Verify `C:\opencv\build\x64\vc16\lib` contains `.lib` files
2. Check that you have the correct OpenCV version for VS2022

### If Visual Studio compiler is not found:
1. Make sure you're using Developer Command Prompt
2. Verify Visual Studio 2022 is installed with C++ workload
3. Try running the VsDevCmd.bat file manually

## Quick Verification Commands

Before building, run these to verify your setup:

```cmd
# Check CMake
cmake --version

# Check Visual Studio compiler
where cl

# Check OpenCV files exist
dir "C:\opencv\build\x64\vc16\lib\opencv_world*.lib"

# Check PortAudio source
dir "C:\Users\KIIT\Downloads\pa_stable_v190700_20210406 (1)\portaudio\include\portaudio.h"
```
1. Open "System Properties" → "Advanced" → "Environment Variables"
2. Under "System Variables", click "New" and add:
   - Variable name: `OPENCV_DIR`
   - Variable value: `C:\opencv\build`
3. Find the "Path" variable in "System Variables" and edit it
4. Add new entry: `C:\opencv\build\x64\vc16\bin`
5. Click "OK" to save

### 3.3 Verify OpenCV Installation
1. Open Command Prompt
2. Navigate to `C:\opencv\build\x64\vc16\bin`
3. You should see files like `opencv_world4xx.dll`

## Step 4: Install PortAudio

### Option A: Using vcpkg (Recommended)

#### 4.1 Install vcpkg
1. Open Command Prompt as Administrator
2. Navigate to C:\ drive:
   ```cmd
   cd C:\
   ```
3. Clone vcpkg:
   ```cmd
   git clone https://github.com/Microsoft/vcpkg.git
   ```
4. Navigate to vcpkg and bootstrap:
   ```cmd
   cd vcpkg
   .\bootstrap-vcpkg.bat
   ```

#### 4.2 Install PortAudio via vcpkg
1. Install PortAudio for x64:
   ```cmd
   .\vcpkg install portaudio:x64-windows
   ```
2. Integrate with Visual Studio:
   ```cmd
   .\vcpkg integrate install
   ```

#### 4.3 Set vcpkg Environment Variable
1. Go to "System Properties" → "Environment Variables"
2. Add new system variable:
   - Variable name: `CMAKE_TOOLCHAIN_FILE`
   - Variable value: `C:\vcpkg\scripts\buildsystems\vcpkg.cmake`

### Option B: Manual Installation (Alternative)

#### 4.1 Download PortAudio
1. Go to: http://files.portaudio.com/download.html
2. Download "PA Stable v19.7.0" → "PortAudio Windows MSVC"
3. Extract to `C:\portaudio`

#### 4.2 Build PortAudio (if prebuilt not available)
1. Open "x64 Native Tools Command Prompt for VS 2022"
2. Navigate to portaudio directory
3. Create build directory and compile:
   ```cmd
   cd C:\portaudio
   mkdir build
   cd build
   cmake .. -A x64
   cmake --build . --config Release
   ```

## Step 5: Install Git (if not already installed)

1. Go to: https://git-scm.com/download/win
2. Download and install Git for Windows
3. Use default settings during installation

## Step 6: Verify All Dependencies

### 6.1 Create Test Script
Create a file `test_dependencies.bat` with this content:

```batch
@echo off
echo Testing Dependencies...
echo.

echo Testing CMake:
cmake --version
if errorlevel 1 (
    echo ERROR: CMake not found
) else (
    echo SUCCESS: CMake found
)
echo.

echo Testing Git:
git --version
if errorlevel 1 (
    echo ERROR: Git not found
) else (
    echo SUCCESS: Git found
)
echo.

echo Testing Visual Studio Compiler:
where cl
if errorlevel 1 (
    echo ERROR: Visual Studio compiler not found
    echo Please run this from "x64 Native Tools Command Prompt for VS 2022"
) else (
    echo SUCCESS: Visual Studio compiler found
)
echo.

echo Testing OpenCV:
if exist "C:\opencv\build\x64\vc16\bin\opencv_world*.dll" (
    echo SUCCESS: OpenCV found
) else (
    echo ERROR: OpenCV not found at C:\opencv\build\x64\vc16\bin\
)
echo.

echo Testing vcpkg (if used):
if exist "C:\vcpkg\vcpkg.exe" (
    echo SUCCESS: vcpkg found
    C:\vcpkg\vcpkg.exe list portaudio
) else (
    echo WARNING: vcpkg not found (may use manual PortAudio installation)
)
echo.

pause
```

### 6.2 Run the Test
1. Save the script to your project folder
2. Open "x64 Native Tools Command Prompt for VS 2022"
3. Navigate to your project folder
4. Run: `test_dependencies.bat`

## Step 7: Build the Project

### 7.1 Open Correct Command Prompt
**Important**: Always use "x64 Native Tools Command Prompt for VS 2022" for building

### 7.2 Navigate and Build
```cmd
cd "C:\Users\KIIT\Downloads\emotion_analyzer_cpp\New folder"
mkdir build
cd build
cmake .. -A x64
cmake --build . --config Release
```

## Troubleshooting Common Issues

### Issue 1: CMake can't find OpenCV
**Solution**: 
- Verify `OPENCV_DIR` environment variable
- Restart Command Prompt after setting environment variables
- Use: `cmake .. -A x64 -DOpenCV_DIR=C:\opencv\build`

### Issue 2: PortAudio not found
**Solution**:
- If using vcpkg: Ensure `CMAKE_TOOLCHAIN_FILE` is set
- If manual install: Use `cmake .. -A x64 -DPORTAUDIO_ROOT=C:\portaudio`

### Issue 3: Compiler not found
**Solution**:
- Use "x64 Native Tools Command Prompt for VS 2022"
- Ensure Visual Studio C++ workload is installed

### Issue 4: Permission errors
**Solution**:
- Run Command Prompt as Administrator
- Ensure antivirus isn't blocking the build

## Environment Variables Summary

After installation, you should have these environment variables:

1. **Path** should include:
   - `C:\Program Files\CMake\bin`
   - `C:\opencv\build\x64\vc16\bin`
   - `C:\Program Files\Git\bin`

2. **System Variables**:
   - `OPENCV_DIR` = `C:\opencv\build`
   - `CMAKE_TOOLCHAIN_FILE` = `C:\vcpkg\scripts\buildsystems\vcpkg.cmake` (if using vcpkg)

## Final Verification

Once everything is installed:

1. Open "x64 Native Tools Command Prompt for VS 2022"
2. Navigate to your project: `cd "C:\Users\KIIT\Downloads\emotion_analyzer_cpp\New folder"`
3. Run the provided build script: `build.bat`
4. If successful, you should see: `emotion_analyzer.exe` in `build\Release\`

## Need Help?

If you encounter issues:
1. Check the error messages carefully
2. Ensure you're using the correct command prompt
3. Verify all environment variables are set
4. Restart your computer after setting environment variables
5. Run the dependency test script to identify missing components

This setup should give you a fully functional development environment for the Multimodal Emotion Analyzer!
