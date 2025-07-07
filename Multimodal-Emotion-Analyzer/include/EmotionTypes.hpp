#pragma once

#include <string>
#include <chrono>
#include <array>

namespace EmotionAnalysis {

enum class EmotionType {
    ANGER = 0,
    DISGUST = 1,
    FEAR = 2,
    HAPPINESS = 3,
    SADNESS = 4,
    SURPRISE = 5,
    NEUTRAL = 6,
    UNKNOWN = 7,
    COUNT = 8
};

struct EmotionResult {
    EmotionType dominant_emotion;
    std::array<float, 8> probabilities; // Probabilities for each emotion
    float confidence;
    std::chrono::steady_clock::time_point timestamp;
    
    EmotionResult() 
        : dominant_emotion(EmotionType::UNKNOWN)
        , probabilities{0.0f}
        , confidence(0.0f)
        , timestamp(std::chrono::steady_clock::now()) {}
};

struct MultimodalResult {
    EmotionResult video_emotion;
    EmotionResult audio_emotion;
    EmotionResult fused_emotion;
    bool is_valid;
    
    MultimodalResult() : is_valid(false) {}
};

// Utility functions
std::string emotionToString(EmotionType emotion);
EmotionType stringToEmotion(const std::string& emotion_str);

} // namespace EmotionAnalysis
