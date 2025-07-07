# How to Run the Multimodal Emotion Analyzer Project

This guide will teach you how to build and run the emotion analyzer project from scratch.

## 📋 Prerequisites

Before running the project, ensure you have the following installed:

### Required Software:
1. **Visual Studio 2022** (Community/Professional/Enterprise)
   - With C++ development workload
   - Windows 10/11 SDK
2. **CMake** (version 3.20 or higher)
3. **Git** (for version control)

### Required Libraries:
1. **OpenCV 4.12.0** (installed at `C:\opencv\`)
2. **PortAudio** (for audio processing)

## 🚀 Quick Start Guide

### Step 1: Navigate to Project Directory
```powershell
# Open PowerShell and navigate to the project
cd "C:\Users\KIIT\Downloads\emotion_analyzer_cpp\New folder\Multimodal-Emotion-Analyzer"
```

### Step 2: Build the Project
```powershell
# Run the build script (this does everything automatically)
.\build.bat
```

### Step 3: Run the Application
```powershell
# Navigate to the executable directory
cd build\Release

# Run with different options:

# 1. Run in test mode (basic functionality test)
emotion_analyzer.exe --test

# 2. Show help and available options
emotion_analyzer.exe --help

# 3. Run with default settings (camera + microphone)
emotion_analyzer.exe

# 4. Run with specific camera ID
emotion_analyzer.exe --camera 1

# 5. Run audio-only emotion analysis
emotion_analyzer.exe --audio-only

# 6. Run video-only emotion analysis
emotion_analyzer.exe --video-only

# 7. Run without video display
emotion_analyzer.exe --no-video
```

## 🎯 Understanding the Application

### What It Does:
- **Video Analysis**: Detects faces and analyzes facial emotions in real-time
- **Audio Analysis**: Captures microphone input and analyzes emotional tone
- **Multimodal Fusion**: Combines video and audio results for better accuracy
- **Real-time Output**: Shows emotion percentages continuously

### Expected Output:
```
=== Multimodal Emotion Analyzer ===
Real-time emotion analysis from video and audio input
Press 'q' or ESC in the video window to quit
=========================================
Starting application...
Configuration complete. Starting analyzer...
Running in full multimodal mode
Emotion Update - Video: Happy (75%), Audio: Neutral (60%), Fused: Happy (68%)
```

## 🔧 Detailed Build Process

### Manual Build Steps (if build.bat doesn't work):

1. **Setup Visual Studio Environment**:
```cmd
# Open Developer Command Prompt for VS 2022, or run:
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

2. **Create and Configure Build Directory**:
```powershell
mkdir build -Force
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
```

3. **Build the Project**:
```powershell
cmake --build . --config Release
```

4. **Run the Application**:
```powershell
cd Release
.\emotion_analyzer.exe
```

## 🎮 Using the Application

### Controls:
- **'q'** or **ESC**: Quit the application
- **Ctrl+C**: Emergency stop (in terminal)

### Camera Options:
- `--camera 0`: Default camera (usually built-in webcam)
- `--camera 1`: Second camera (if available)
- `--width 1280 --height 720`: Set video resolution

### Processing Modes:
- **Full Mode** (default): Video + Audio analysis
- **Audio Only**: `--audio-only` - Just microphone analysis
- **Video Only**: `--video-only` - Just camera analysis

## 🔍 Troubleshooting

### Common Issues:

1. **"DLL not found" error**:
   ```powershell
   # Make sure you're in the Release directory with all DLLs
   cd build\Release
   dir *.dll  # Should show OpenCV and PortAudio DLLs
   ```

2. **Camera not detected**:
   ```powershell
   # Try different camera IDs
   emotion_analyzer.exe --camera 0
   emotion_analyzer.exe --camera 1
   ```

3. **Build fails**:
   ```powershell
   # Clean and rebuild
   rmdir build -Recurse -Force
   .\build.bat
   ```

4. **No audio input**:
   - Check microphone permissions in Windows Settings
   - Try `--video-only` mode to test video separately

### Verify Installation:
```powershell
# Test basic functionality
emotion_analyzer.exe --test

# Check OpenCV version
emotion_analyzer.exe --test | findstr "OpenCV"
```

## 📊 Understanding the Output

### Emotion Types Detected:
- **Happy**: Smiling, positive expressions
- **Sad**: Frowning, downward expressions  
- **Angry**: Furrowed brow, tense expressions
- **Surprise**: Wide eyes, raised eyebrows
- **Fear**: Worried, anxious expressions
- **Disgust**: Nose wrinkled, negative expressions
- **Neutral**: Calm, no strong emotion

### Confidence Levels:
- **0-30%**: Low confidence
- **31-60%**: Medium confidence  
- **61-100%**: High confidence

### Fusion Strategy:
The application combines video and audio emotions using weighted averaging, giving more weight to the modality with higher confidence.

## 🔄 Development Workflow

### For Developers:

1. **Edit Source Code**: Modify files in `src/` and `include/`
2. **Rebuild**: Run `.\build.bat`
3. **Test**: Run `emotion_analyzer.exe --test`
4. **Debug**: Use VS Code with the included configuration

### Project Structure:
```
src/                    # Source files to modify
├── main.cpp           # Application entry point
├── VideoEmotionAnalyzer_improved.cpp  # Video processing
├── AudioEmotionAnalyzer.cpp           # Audio processing
└── MultimodalEmotionFusion.cpp        # Fusion logic

include/               # Header files
models/               # AI models (don't modify)
build/Release/        # Final executable and DLLs
```

## 🎯 Performance Tips

### For Best Results:
1. **Good Lighting**: Ensure face is well-lit for video analysis
2. **Clear Audio**: Speak clearly near microphone for audio analysis
3. **Stable Camera**: Minimize camera movement for better face detection
4. **Close Application Properly**: Use 'q' or ESC, not force-close

### System Requirements:
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **Camera**: Any USB/built-in webcam
- **Microphone**: Any audio input device
- **OS**: Windows 10/11 (64-bit)

## 🎉 Success Indicators

You know the project is working correctly when you see:
- ✅ Camera window opens showing your face with detection box
- ✅ Real-time emotion percentages updating in terminal
- ✅ Both video and audio emotions being detected
- ✅ Fused emotion results showing combined analysis
- ✅ No error messages or crashes

Happy emotion analyzing! 🎭
