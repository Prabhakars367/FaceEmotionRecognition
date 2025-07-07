#pragma once

#include "EmotionTypes.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace EmotionAnalysis {

/**
 * @brief Python-based deep learning emotion analyzer using pybind11
 * 
 * This class bridges C++ with Python deep learning models for emotion recognition.
 * It provides high-accuracy emotion detection using pre-trained neural networks
 * while maintaining real-time performance through efficient data exchange.
 */
class PythonEmotionInference {
public:
    PythonEmotionInference();
    ~PythonEmotionInference();

    /**
     * @brief Initialize the Python interpreter and load the emotion model
     * @param model_path Path to the Python emotion model (.h5, .pt, .onnx)
     * @param python_script_path Path to the Python inference script
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& model_path, const std::string& python_script_path = "");

    /**
     * @brief Perform emotion inference on a preprocessed face image
     * @param face_image Preprocessed face image (typically 48x48 or 224x224)
     * @return EmotionResult with probabilities and dominant emotion
     */
    EmotionResult predictEmotion(const cv::Mat& face_image);

    /**
     * @brief Batch process multiple face images for efficiency
     * @param face_images Vector of preprocessed face images
     * @return Vector of EmotionResult for each input image
     */
    std::vector<EmotionResult> predictEmotionsBatch(const std::vector<cv::Mat>& face_images);

    /**
     * @brief Check if the Python model is loaded and ready
     * @return true if model is ready for inference
     */
    bool isModelLoaded() const;

    /**
     * @brief Get the expected input size for the loaded model
     * @return cv::Size containing width and height requirements
     */
    cv::Size getModelInputSize() const;

    /**
     * @brief Preprocess face image according to model requirements
     * @param face_image Input face image
     * @return Preprocessed image ready for model inference
     */
    cv::Mat preprocessForModel(const cv::Mat& face_image);

    /**
     * @brief Set confidence threshold for emotion detection
     * @param threshold Minimum confidence threshold (0.0 to 1.0)
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief Get available emotion classes from the model
     * @return Vector of emotion class names
     */
    std::vector<std::string> getEmotionClasses() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

/**
 * @brief Factory function to create Python emotion inference instance
 * @param model_type Type of model ("tensorflow", "pytorch", "onnx")
 * @return Unique pointer to PythonEmotionInference instance
 */
std::unique_ptr<PythonEmotionInference> createPythonInference(const std::string& model_type = "tensorflow");

} // namespace EmotionAnalysis
