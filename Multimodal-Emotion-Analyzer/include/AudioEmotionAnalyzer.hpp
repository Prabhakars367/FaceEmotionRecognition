#pragma once

#include "EmotionTypes.hpp"
#include <vector>
#include <memory>
#include <functional>

#ifndef NO_AUDIO
#include <portaudio.h>
#endif

namespace EmotionAnalysis {

struct AudioConfig {
    int sample_rate = 16000;
    int channels = 1;
    int frames_per_buffer = 512;
    double duration_seconds = 0.5; // Analysis window duration (reduced for more responsiveness)
};

class AudioEmotionAnalyzer {
public:
    AudioEmotionAnalyzer();
    ~AudioEmotionAnalyzer();
    
    // Initialize audio system
    bool initialize(const AudioConfig& config = AudioConfig{});
    
    // Start/stop audio capture
    bool startCapture();
    bool stopCapture();
    
    // Get the latest emotion analysis result
    EmotionResult getLatestResult() const;
    
    // Set callback for real-time emotion updates
    void setEmotionCallback(std::function<void(const EmotionResult&)> callback);
    
    // Manual analysis of audio buffer
    EmotionResult analyzeAudioBuffer(const std::vector<float>& audio_data);
    
    // Get current audio level (for visualization)
    float getCurrentAudioLevel() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Audio processing methods
    std::vector<float> extractFeatures(const std::vector<float>& audio_data);
    EmotionResult classifyAudioEmotion(const std::vector<float>& features);
    
    // Feature extraction helpers
    float calculateZeroCrossingRate(const std::vector<float>& audio);
    float calculateSpectralCentroid(const std::vector<float>& audio);
    std::vector<float> calculateMFCC(const std::vector<float>& audio);
};

} // namespace EmotionAnalysis
