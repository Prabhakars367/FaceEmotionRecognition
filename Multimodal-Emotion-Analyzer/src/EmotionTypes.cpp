#include "EmotionTypes.hpp"
#include <unordered_map>

namespace EmotionAnalysis {

std::string emotionToString(EmotionType emotion) {
    static const std::unordered_map<EmotionType, std::string> emotion_map = {
        {EmotionType::ANGER, "Angry"},
        {EmotionType::DISGUST, "Disgust"},
        {EmotionType::FEAR, "Fear"},
        {EmotionType::HAPPINESS, "Happy"},
        {EmotionType::SADNESS, "Sad"},
        {EmotionType::SURPRISE, "Surprise"},
        {EmotionType::NEUTRAL, "Neutral"},
        {EmotionType::UNKNOWN, "Unknown"}
    };
    
    auto it = emotion_map.find(emotion);
    return (it != emotion_map.end()) ? it->second : "Unknown";
}

EmotionType stringToEmotion(const std::string& emotion_str) {
    static const std::unordered_map<std::string, EmotionType> string_map = {
        {"Angry", EmotionType::ANGER},
        {"Disgust", EmotionType::DISGUST},
        {"Fear", EmotionType::FEAR},
        {"Happy", EmotionType::HAPPINESS},
        {"Sad", EmotionType::SADNESS},
        {"Surprise", EmotionType::SURPRISE},
        {"Neutral", EmotionType::NEUTRAL}
    };
    
    auto it = string_map.find(emotion_str);
    return (it != string_map.end()) ? it->second : EmotionType::UNKNOWN;
}

} // namespace EmotionAnalysis
