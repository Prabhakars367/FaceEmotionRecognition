#pragma once

#include "VideoEmotionAnalyzer.hpp"
#include "AudioEmotionAnalyzer.hpp"
#include "MultimodalEmotionFusion.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>

namespace EmotionAnalysis {

struct AppConfig {
    // Video settings
    int camera_id = 0;
    int video_width = 640;
    int video_height = 480;
    int video_fps = 30;
    
    // Audio settings
    AudioConfig audio_config;
    
    // Fusion settings
    FusionConfig fusion_config;
    
    // Display settings
    bool show_video = true;
    bool show_audio_level = true;
    bool show_probabilities = true;
    int update_interval_ms = 100; // Update frequency for emotions
};

class MultimodalEmotionAnalyzer {
public:
    MultimodalEmotionAnalyzer();
    ~MultimodalEmotionAnalyzer();
    
    // Initialize the complete system
    bool initialize(const AppConfig& config = AppConfig{});
    
    // Start real-time analysis
    bool start();
    
    // Stop analysis
    void stop();
    
    // Check if system is running
    bool isRunning() const;
    
    // Get latest multimodal result
    MultimodalResult getLatestResult() const;
    
    // Set callback for emotion updates
    void setEmotionCallback(std::function<void(const MultimodalResult&)> callback);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Main processing loops
    void videoProcessingLoop();
    void audioProcessingLoop();
    void fusionLoop();
    void displayLoop();
    
    // Helper methods
    void drawEmotionOverlay(cv::Mat& frame, const MultimodalResult& result);
    void drawAudioLevel(cv::Mat& frame, float audio_level);
    void drawProbabilities(cv::Mat& frame, const EmotionResult& result, const cv::Point& position);
    std::string formatEmotionText(const EmotionResult& result);
};

} // namespace EmotionAnalysis
