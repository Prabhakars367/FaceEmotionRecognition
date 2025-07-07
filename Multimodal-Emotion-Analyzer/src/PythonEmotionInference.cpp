#include "PythonEmotionInference.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <filesystem>

// Include pybind11 headers
#ifdef PYTHON_INTEGRATION_ENABLED
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

namespace EmotionAnalysis {

class PythonEmotionInference::Impl {
public:
    bool is_initialized = false;
    bool model_loaded = false;
    cv::Size model_input_size{48, 48}; // Default size, will be updated based on model
    float confidence_threshold = 0.3f;
    std::vector<std::string> emotion_classes;
    
#ifdef PYTHON_INTEGRATION_ENABLED
    py::object python_module;
    py::object emotion_model;
    py::object predict_function;
    py::scoped_interpreter python_interpreter;
#endif

    // Model preprocessing parameters
    bool normalize_input = true;
    bool grayscale_input = true;
    cv::Scalar mean_values{0.485, 0.456, 0.406}; // ImageNet defaults
    cv::Scalar std_values{0.229, 0.224, 0.225};
    
    Impl() {
        // Initialize default emotion classes
        emotion_classes = {
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
        };
    }
    
    bool initializePython(const std::string& model_path, const std::string& script_path) {
#ifdef PYTHON_INTEGRATION_ENABLED
        try {
            // Import required Python modules
            py::module sys = py::module::import("sys");
            py::module os = py::module::import("os");
            
            // Add current directory to Python path
            std::string current_dir = std::filesystem::current_path().string();
            sys.attr("path").attr("insert")(0, current_dir);
            
            // Import deep learning framework modules
            py::module np = py::module::import("numpy");
            py::module cv2 = py::module::import("cv2");
            
            // Try to import TensorFlow/Keras
            try {
                py::module tf = py::module::import("tensorflow");
                py::module keras = py::module::import("tensorflow.keras");
                
                // Load the emotion recognition model
                auto load_model = keras.attr("models").attr("load_model");
                emotion_model = load_model(model_path);
                
                // Get model input shape
                auto input_shape = emotion_model.attr("input_shape");
                auto shape_tuple = input_shape.cast<py::tuple>();
                if (shape_tuple.size() >= 3) {
                    model_input_size = cv::Size(
                        shape_tuple[2].cast<int>(), 
                        shape_tuple[1].cast<int>()
                    );
                }
                
                model_loaded = true;
                std::cout << "TensorFlow emotion model loaded successfully from: " << model_path << std::endl;
                std::cout << "Model input size: " << model_input_size.width << "x" << model_input_size.height << std::endl;
                
            } catch (const py::error_already_set& e) {
                std::cerr << "Failed to load TensorFlow model: " << e.what() << std::endl;
                
                // Try PyTorch as fallback
                try {
                    py::module torch = py::module::import("torch");
                    emotion_model = torch.attr("load")(model_path);
                    model_loaded = true;
                    std::cout << "PyTorch emotion model loaded successfully" << std::endl;
                } catch (const py::error_already_set& e2) {
                    std::cerr << "Failed to load PyTorch model: " << e2.what() << std::endl;
                    return false;
                }
            }
            
            is_initialized = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Error initializing Python emotion inference: " << e.what() << std::endl;
            return false;
        }
#else
        std::cerr << "Python integration not enabled. Please compile with PYTHON_INTEGRATION_ENABLED." << std::endl;
        return false;
#endif
    }
    
    EmotionResult performInference(const cv::Mat& preprocessed_image) {
        EmotionResult result;
        
#ifdef PYTHON_INTEGRATION_ENABLED
        if (!model_loaded) {
            std::cerr << "Model not loaded" << std::endl;
            return result;
        }
        
        try {
            // Convert OpenCV Mat to numpy array
            py::array_t<float> np_array = matToNumpyArray(preprocessed_image);
            
            // Add batch dimension if needed
            if (np_array.ndim() == 3) {
                np_array = np_array.attr("reshape")(py::make_tuple(1, np_array.shape(0), np_array.shape(1), np_array.shape(2)));
            } else if (np_array.ndim() == 2) {
                np_array = np_array.attr("reshape")(py::make_tuple(1, np_array.shape(0), np_array.shape(1), 1));
            }
            
            // Perform prediction
            auto predictions = emotion_model.attr("predict")(np_array);
            auto pred_array = predictions.cast<py::array_t<float>>();
            
            // Extract probabilities
            auto pred_ptr = static_cast<float*>(pred_array.mutable_unchecked<2>().mutable_data(0, 0));
            
            // Fill result probabilities
            for (int i = 0; i < static_cast<int>(EmotionType::COUNT) && i < pred_array.shape(1); ++i) {
                result.probabilities[i] = pred_ptr[i];
            }
            
            // Find dominant emotion
            auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
            int max_index = std::distance(result.probabilities.begin(), max_it);
            
            result.dominant_emotion = static_cast<EmotionType>(max_index);
            result.confidence = *max_it;
            result.timestamp = std::chrono::steady_clock::now();
            
            // Apply confidence threshold
            if (result.confidence < confidence_threshold) {
                result.dominant_emotion = EmotionType::NEUTRAL;
                result.confidence *= 0.7f; // Reduce confidence for uncertain predictions
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error during Python inference: " << e.what() << std::endl;
        }
#endif
        
        return result;
    }
    
private:
#ifdef PYTHON_INTEGRATION_ENABLED
    py::array_t<float> matToNumpyArray(const cv::Mat& mat) {
        cv::Mat float_mat;
        mat.convertTo(float_mat, CV_32F);
        
        if (normalize_input) {
            float_mat /= 255.0f;
        }
        
        return py::array_t<float>(
            {float_mat.rows, float_mat.cols, float_mat.channels()},
            {sizeof(float) * float_mat.cols * float_mat.channels(), sizeof(float) * float_mat.channels(), sizeof(float)},
            float_mat.ptr<float>(),
            py::cast(float_mat)
        );
    }
#endif
};

// PythonEmotionInference implementation
PythonEmotionInference::PythonEmotionInference() 
    : pImpl(std::make_unique<Impl>()) {}

PythonEmotionInference::~PythonEmotionInference() = default;

bool PythonEmotionInference::initialize(const std::string& model_path, const std::string& python_script_path) {
    return pImpl->initializePython(model_path, python_script_path);
}

EmotionResult PythonEmotionInference::predictEmotion(const cv::Mat& face_image) {
    if (!pImpl->model_loaded) {
        return EmotionResult{};
    }
    
    cv::Mat preprocessed = preprocessForModel(face_image);
    return pImpl->performInference(preprocessed);
}

std::vector<EmotionResult> PythonEmotionInference::predictEmotionsBatch(const std::vector<cv::Mat>& face_images) {
    std::vector<EmotionResult> results;
    results.reserve(face_images.size());
    
    for (const auto& image : face_images) {
        results.push_back(predictEmotion(image));
    }
    
    return results;
}

bool PythonEmotionInference::isModelLoaded() const {
    return pImpl->model_loaded;
}

cv::Size PythonEmotionInference::getModelInputSize() const {
    return pImpl->model_input_size;
}

cv::Mat PythonEmotionInference::preprocessForModel(const cv::Mat& face_image) {
    cv::Mat processed;
    
    // Resize to model input size
    cv::resize(face_image, processed, pImpl->model_input_size);
    
    // Convert to grayscale if required
    if (pImpl->grayscale_input && processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
    }
    
    // Convert to float and normalize
    processed.convertTo(processed, CV_32F);
    
    if (pImpl->normalize_input) {
        processed /= 255.0f;
    }
    
    // Apply mean and std normalization if specified
    if (processed.channels() == 3) {
        cv::subtract(processed, pImpl->mean_values, processed);
        cv::divide(processed, pImpl->std_values, processed);
    }
    
    return processed;
}

void PythonEmotionInference::setConfidenceThreshold(float threshold) {
    pImpl->confidence_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

std::vector<std::string> PythonEmotionInference::getEmotionClasses() const {
    return pImpl->emotion_classes;
}

// Factory function
std::unique_ptr<PythonEmotionInference> createPythonInference(const std::string& model_type) {
    auto inference = std::make_unique<PythonEmotionInference>();
    
    // Additional configuration based on model type
    if (model_type == "tensorflow" || model_type == "keras") {
        // TensorFlow/Keras specific settings
        std::cout << "Creating TensorFlow-based emotion inference" << std::endl;
    } else if (model_type == "pytorch") {
        // PyTorch specific settings
        std::cout << "Creating PyTorch-based emotion inference" << std::endl;
    } else if (model_type == "onnx") {
        // ONNX specific settings
        std::cout << "Creating ONNX-based emotion inference" << std::endl;
    }
    
    return inference;
}

} // namespace EmotionAnalysis
