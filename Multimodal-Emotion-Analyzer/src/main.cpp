#include "MultimodalEmotionAnalyzer.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

using namespace EmotionAnalysis;

// Global analyzer instance for signal handling
std::unique_ptr<MultimodalEmotionAnalyzer> g_analyzer;
std::atomic<bool> g_should_exit{false};

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Shutting down gracefully..." << std::endl;
    g_should_exit = true;
    if (g_analyzer) {
        g_analyzer->stop();
    }
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -h, --help          Show this help message\n"
              << "  -c, --camera ID     Camera ID (default: 0)\n"
              << "  -w, --width WIDTH   Video width (default: 640)\n"
              << "  -h, --height HEIGHT Video height (default: 480)\n"
              << "  --no-video          Disable video display\n"
              << "  --audio-only        Audio emotion analysis only\n"
              << "  --video-only        Video emotion analysis only\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== Multimodal Emotion Analyzer ===" << std::endl;
    std::cout << "Real-time emotion analysis from video and audio input" << std::endl;
    std::cout << "Press 'q' or ESC in the video window to quit" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Setup signal handling
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Parse command line arguments
    AppConfig config;
    bool show_help = false;
    bool audio_only = false;
    bool video_only = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        }
        else if (arg == "-c" || arg == "--camera") {
            if (i + 1 < argc) {
                config.camera_id = std::atoi(argv[++i]);
            }
        }
        else if (arg == "-w" || arg == "--width") {
            if (i + 1 < argc) {
                config.video_width = std::atoi(argv[++i]);
            }
        }
        else if (arg == "--height") {
            if (i + 1 < argc) {
                config.video_height = std::atoi(argv[++i]);
            }
        }
        else if (arg == "--no-video") {
            config.show_video = false;
        }
        else if (arg == "--audio-only") {
            audio_only = true;
        }
        else if (arg == "--video-only") {
            video_only = true;
        }
    }
    
    if (show_help) {
        printUsage(argv[0]);
        return 0;
    }
    
    // Adjust config based on mode
    if (audio_only) {
        config.show_video = false;
        config.fusion_config.video_weight = 0.0f;
        config.fusion_config.audio_weight = 1.0f;
    }
    else if (video_only) {
        config.fusion_config.video_weight = 1.0f;
        config.fusion_config.audio_weight = 0.0f;
    }
    
    try {
        // Create and initialize analyzer
        g_analyzer = std::make_unique<MultimodalEmotionAnalyzer>();
        
        if (!g_analyzer->initialize(config)) {
            std::cerr << "Failed to initialize emotion analyzer" << std::endl;
            return -1;
        }
        
        // Set up emotion callback for logging
        g_analyzer->setEmotionCallback([](const MultimodalResult& result) {
            if (result.is_valid) {
                std::cout << "Emotion Update - "
                          << "Video: " << emotionToString(result.video_emotion.dominant_emotion) 
                          << " (" << (result.video_emotion.confidence * 100) << "%), "
                          << "Audio: " << emotionToString(result.audio_emotion.dominant_emotion)
                          << " (" << (result.audio_emotion.confidence * 100) << "%), "
                          << "Fused: " << emotionToString(result.fused_emotion.dominant_emotion)
                          << " (" << (result.fused_emotion.confidence * 100) << "%)"
                          << std::endl;
            }
        });
        
        // Start analysis
        if (!g_analyzer->start()) {
            std::cerr << "Failed to start emotion analyzer" << std::endl;
            return -1;
        }
        
        // Main loop
        while (!g_should_exit && g_analyzer->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Clean shutdown
        std::cout << "Shutting down..." << std::endl;
        g_analyzer->stop();
        g_analyzer.reset();
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Goodbye!" << std::endl;
    return 0;
}
