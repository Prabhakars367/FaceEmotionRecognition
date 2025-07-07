#pragma once

#include "EmotionTypes.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace EmotionAnalysis {

class VideoEmotionAnalyzer {
public:
    VideoEmotionAnalyzer();
    ~VideoEmotionAnalyzer();
    
    // Initialize the analyzer with model paths
    bool initialize(const std::string& face_cascade_path, 
                   const std::string& emotion_model_path = "");
    
    // Analyze emotion from a frame
    EmotionResult analyzeFrame(const cv::Mat& frame);
    
    // Get detected faces from the last frame
    std::vector<cv::Rect> getLastDetectedFaces() const;
    
    // Set minimum face size for detection
    void setMinFaceSize(int min_size);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Helper methods
    cv::Mat preprocessFace(const cv::Mat& face);
    EmotionResult classifyEmotion(const cv::Mat& preprocessed_face);
    EmotionResult analyzeFacialFeatures(const cv::Mat& face);
};

} // namespace EmotionAnalysis
