<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Multimodal Emotion Analyzer Project Instructions

This is a C++ project for real-time multimodal emotion analysis combining video and audio input.

## Code Style Guidelines
- Use modern C++20 features where appropriate
- Follow RAII principles for resource management
- Use smart pointers instead of raw pointers
- Implement proper error handling with exceptions or error codes
- Write self-documenting code with clear variable and function names

## Architecture Guidelines
- Keep video and audio processing modules separate but coordinated
- Use observer pattern for emotion detection events
- Implement thread-safe queues for real-time data processing
- Design for extensibility - new emotion models should be easy to integrate

## Dependencies
- OpenCV for computer vision operations
- PortAudio for cross-platform audio I/O
- Modern CMake for build system
- Consider lightweight ML inference libraries (ONNX Runtime, TensorFlow Lite)

## Performance Considerations
- Optimize for real-time processing (< 100ms latency)
- Use efficient data structures for video frames and audio buffers
- Implement proper memory management to avoid leaks
- Consider multi-threading for parallel video/audio processing
