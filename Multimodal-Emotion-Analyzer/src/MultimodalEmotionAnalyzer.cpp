#include "MultimodalEmotionAnalyzer.hpp"
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace EmotionAnalysis {

class MultimodalEmotionAnalyzer::Impl {
public:
    AppConfig config;
    
    // Component instances
    VideoEmotionAnalyzer video_analyzer;
    AudioEmotionAnalyzer audio_analyzer;
    MultimodalEmotionFusion fusion;
    
    // OpenCV video capture
    cv::VideoCapture camera;
    
    // Threading
    std::atomic<bool> is_running{false};
    std::atomic<bool> should_stop{false};
    
    std::thread video_thread;
    std::thread audio_thread;
    std::thread fusion_thread;
    std::thread display_thread;
    
    // Shared data with thread safety
    std::mutex video_result_mutex;
    std::mutex audio_result_mutex;
    std::mutex multimodal_result_mutex;
    
    EmotionResult latest_video_result;
    EmotionResult latest_audio_result;
    MultimodalResult latest_multimodal_result;
    
    // Callback for emotion updates
    std::function<void(const MultimodalResult&)> emotion_callback;
    
    // Timing
    std::chrono::steady_clock::time_point last_update_time;
};

MultimodalEmotionAnalyzer::MultimodalEmotionAnalyzer() 
    : pImpl(std::make_unique<Impl>()) {}

MultimodalEmotionAnalyzer::~MultimodalEmotionAnalyzer() {
    stop();
}

bool MultimodalEmotionAnalyzer::initialize(const AppConfig& config) {
    pImpl->config = config;
    
    try {
        // Initialize video analyzer
        std::string cascade_path = "models/haarcascade_frontalface_default.xml";
        if (!pImpl->video_analyzer.initialize(cascade_path)) {
            std::cerr << "Failed to initialize video emotion analyzer" << std::endl;
            return false;
        }
        
        // Initialize audio analyzer
        if (!pImpl->audio_analyzer.initialize(config.audio_config)) {
            std::cerr << "Failed to initialize audio emotion analyzer" << std::endl;
            return false;
        }
        
        // Initialize fusion engine
        pImpl->fusion.initialize(config.fusion_config);
        
        // Initialize camera
        pImpl->camera.open(config.camera_id);
        if (!pImpl->camera.isOpened()) {
            std::cerr << "Failed to open camera " << config.camera_id << std::endl;
            return false;
        }
        
        // Set camera properties
        pImpl->camera.set(cv::CAP_PROP_FRAME_WIDTH, config.video_width);
        pImpl->camera.set(cv::CAP_PROP_FRAME_HEIGHT, config.video_height);
        pImpl->camera.set(cv::CAP_PROP_FPS, config.video_fps);
        
        std::cout << "Multimodal emotion analyzer initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error initializing multimodal analyzer: " << e.what() << std::endl;
        return false;
    }
}

bool MultimodalEmotionAnalyzer::start() {
    if (pImpl->is_running) {
        std::cout << "Analyzer is already running" << std::endl;
        return true;
    }
    
    pImpl->should_stop = false;
    pImpl->is_running = true;
    pImpl->last_update_time = std::chrono::steady_clock::now();
    
    try {
        // Start audio capture
        if (!pImpl->audio_analyzer.startCapture()) {
            std::cerr << "Failed to start audio capture" << std::endl;
            return false;
        }
        
        // Start processing threads
        pImpl->video_thread = std::thread(&MultimodalEmotionAnalyzer::videoProcessingLoop, this);
        pImpl->audio_thread = std::thread(&MultimodalEmotionAnalyzer::audioProcessingLoop, this);
        pImpl->fusion_thread = std::thread(&MultimodalEmotionAnalyzer::fusionLoop, this);
        
        if (pImpl->config.show_video) {
            pImpl->display_thread = std::thread(&MultimodalEmotionAnalyzer::displayLoop, this);
        }
        
        std::cout << "Multimodal emotion analysis started" << std::endl;
        std::cout << "Press 'q' or ESC to quit" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error starting analyzer: " << e.what() << std::endl;
        stop();
        return false;
    }
}

void MultimodalEmotionAnalyzer::stop() {
    if (!pImpl->is_running) {
        return;
    }
    
    std::cout << "Stopping multimodal emotion analyzer..." << std::endl;
    
    pImpl->should_stop = true;
    pImpl->is_running = false;
    
    // Stop audio capture
    pImpl->audio_analyzer.stopCapture();
    
    // Wait for threads to finish
    if (pImpl->video_thread.joinable()) {
        pImpl->video_thread.join();
    }
    if (pImpl->audio_thread.joinable()) {
        pImpl->audio_thread.join();
    }
    if (pImpl->fusion_thread.joinable()) {
        pImpl->fusion_thread.join();
    }
    if (pImpl->display_thread.joinable()) {
        pImpl->display_thread.join();
    }
    
    // Release camera
    if (pImpl->camera.isOpened()) {
        pImpl->camera.release();
    }
    
    // Close any OpenCV windows
    cv::destroyAllWindows();
    
    std::cout << "Multimodal emotion analyzer stopped" << std::endl;
}

bool MultimodalEmotionAnalyzer::isRunning() const {
    return pImpl->is_running;
}

MultimodalResult MultimodalEmotionAnalyzer::getLatestResult() const {
    std::lock_guard<std::mutex> lock(pImpl->multimodal_result_mutex);
    return pImpl->latest_multimodal_result;
}

void MultimodalEmotionAnalyzer::setEmotionCallback(std::function<void(const MultimodalResult&)> callback) {
    pImpl->emotion_callback = std::move(callback);
}

void MultimodalEmotionAnalyzer::videoProcessingLoop() {
    cv::Mat frame;
    
    while (!pImpl->should_stop && pImpl->camera.isOpened()) {
        try {
            if (pImpl->camera.read(frame) && !frame.empty()) {
                // Analyze emotion in the frame
                EmotionResult result = pImpl->video_analyzer.analyzeFrame(frame);
                
                // Update shared result
                {
                    std::lock_guard<std::mutex> lock(pImpl->video_result_mutex);
                    pImpl->latest_video_result = result;
                }
            }
            
            // Control frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
            
        } catch (const std::exception& e) {
            std::cerr << "Error in video processing: " << e.what() << std::endl;
        }
    }
}

void MultimodalEmotionAnalyzer::audioProcessingLoop() {
    auto last_analysis_time = std::chrono::steady_clock::now();
    const auto analysis_interval = std::chrono::milliseconds(500); // Analyze every 500ms
    
    while (!pImpl->should_stop) {
        try {
            auto current_time = std::chrono::steady_clock::now();
            
            if (current_time - last_analysis_time >= analysis_interval) {
                // Get latest audio result from the analyzer
                EmotionResult result = pImpl->audio_analyzer.getLatestResult();
                
                // Update shared result if we have a valid result
                if (result.dominant_emotion != EmotionType::UNKNOWN) {
                    std::lock_guard<std::mutex> lock(pImpl->audio_result_mutex);
                    pImpl->latest_audio_result = result;
                }
                
                last_analysis_time = current_time;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in audio processing: " << e.what() << std::endl;
        }
    }
}

void MultimodalEmotionAnalyzer::fusionLoop() {
    while (!pImpl->should_stop) {
        try {
            EmotionResult video_result;
            EmotionResult audio_result;
            
            // Get latest results
            {
                std::lock_guard<std::mutex> lock(pImpl->video_result_mutex);
                video_result = pImpl->latest_video_result;
            }
            {
                std::lock_guard<std::mutex> lock(pImpl->audio_result_mutex);
                audio_result = pImpl->latest_audio_result;
            }
            
            // Perform fusion
            MultimodalResult multimodal_result = 
                pImpl->fusion.createMultimodalResult(video_result, audio_result);
            
            // Update shared result
            {
                std::lock_guard<std::mutex> lock(pImpl->multimodal_result_mutex);
                pImpl->latest_multimodal_result = multimodal_result;
            }
            
            // Call callback if set
            if (pImpl->emotion_callback) {
                pImpl->emotion_callback(multimodal_result);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(pImpl->config.update_interval_ms));
            
        } catch (const std::exception& e) {
            std::cerr << "Error in fusion processing: " << e.what() << std::endl;
        }
    }
}

void MultimodalEmotionAnalyzer::displayLoop() {
    cv::Mat frame;
    
    while (!pImpl->should_stop && pImpl->camera.isOpened()) {
        try {
            if (pImpl->camera.read(frame) && !frame.empty()) {
                // Get latest results
                MultimodalResult result = getLatestResult();
                
                // Draw visualization overlays
                drawEmotionOverlay(frame, result);
                
                if (pImpl->config.show_audio_level) {
                    float audio_level = pImpl->audio_analyzer.getCurrentAudioLevel();
                    drawAudioLevel(frame, audio_level);
                }
                
                // Show the frame
                cv::imshow("Multimodal Emotion Analyzer", frame);
                
                // Check for exit keys
                int key = cv::waitKey(1) & 0xFF;
                if (key == 'q' || key == 27) { // 'q' or ESC
                    pImpl->should_stop = true;
                    break;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error in display loop: " << e.what() << std::endl;
        }
    }
}

void MultimodalEmotionAnalyzer::drawEmotionOverlay(cv::Mat& frame, const MultimodalResult& result) {
    if (!result.is_valid) return;
    
    // Draw face rectangles
    auto faces = pImpl->video_analyzer.getLastDetectedFaces();
    for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        
        // Draw emotion text near the face
        std::string emotion_text = formatEmotionText(result.fused_emotion);
        cv::putText(frame, emotion_text, 
                   cv::Point(face.x, face.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
    
    // Draw overall emotion information
    int y_offset = 30;
    cv::Scalar color_video(255, 0, 0);  // Blue
    cv::Scalar color_audio(0, 255, 0);  // Green
    cv::Scalar color_fused(0, 0, 255);  // Red
    
    // Video emotion
    std::string video_text = "Video: " + formatEmotionText(result.video_emotion);
    cv::putText(frame, video_text, cv::Point(10, y_offset), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, color_video, 2);
    
    // Audio emotion
    y_offset += 25;
    std::string audio_text = "Audio: " + formatEmotionText(result.audio_emotion);
    cv::putText(frame, audio_text, cv::Point(10, y_offset), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, color_audio, 2);
    
    // Fused emotion
    y_offset += 25;
    std::string fused_text = "Fused: " + formatEmotionText(result.fused_emotion);
    cv::putText(frame, fused_text, cv::Point(10, y_offset), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, color_fused, 2);
    
    // Draw probability bars if enabled
    if (pImpl->config.show_probabilities) {
        y_offset += 40;
        drawProbabilities(frame, result.fused_emotion, cv::Point(10, y_offset));
    }
}

void MultimodalEmotionAnalyzer::drawAudioLevel(cv::Mat& frame, float audio_level) {
    // Draw audio level meter
    int meter_width = 200;
    int meter_height = 10;
    int meter_x = frame.cols - meter_width - 10;
    int meter_y = 10;
    
    // Background
    cv::rectangle(frame, 
                 cv::Point(meter_x, meter_y), 
                 cv::Point(meter_x + meter_width, meter_y + meter_height),
                 cv::Scalar(50, 50, 50), -1);
    
    // Level bar
    int level_width = static_cast<int>(meter_width * std::min(1.0f, audio_level * 10)); // Scale up for visibility
    cv::Scalar level_color = (level_width > meter_width * 0.8) ? 
                            cv::Scalar(0, 0, 255) :  // Red for high level
                            cv::Scalar(0, 255, 0);   // Green for normal level
    
    if (level_width > 0) {
        cv::rectangle(frame, 
                     cv::Point(meter_x, meter_y), 
                     cv::Point(meter_x + level_width, meter_y + meter_height),
                     level_color, -1);
    }
    
    // Label
    cv::putText(frame, "Audio Level", cv::Point(meter_x, meter_y - 5), 
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

void MultimodalEmotionAnalyzer::drawProbabilities(cv::Mat& frame, const EmotionResult& result, const cv::Point& position) {
    const std::vector<std::string> emotion_names = {
        "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
    };
    
    int bar_width = 100;
    int bar_height = 12;
    int spacing = 15;
    
    for (size_t i = 0; i < emotion_names.size() && i < result.probabilities.size(); ++i) {
        int y = position.y + static_cast<int>(i * spacing);
        
        // Background bar
        cv::rectangle(frame, 
                     cv::Point(position.x, y), 
                     cv::Point(position.x + bar_width, y + bar_height),
                     cv::Scalar(50, 50, 50), -1);
        
        // Probability bar
        int prob_width = static_cast<int>(bar_width * result.probabilities[i]);
        if (prob_width > 0) {
            cv::Scalar bar_color = (i == static_cast<size_t>(result.dominant_emotion)) ? 
                                  cv::Scalar(0, 0, 255) :  // Red for dominant
                                  cv::Scalar(100, 100, 100); // Gray for others
            
            cv::rectangle(frame, 
                         cv::Point(position.x, y), 
                         cv::Point(position.x + prob_width, y + bar_height),
                         bar_color, -1);
        }
        
        // Label
        std::string label = emotion_names[i] + ": " + 
                           std::to_string(static_cast<int>(result.probabilities[i] * 100)) + "%";
        cv::putText(frame, label, cv::Point(position.x + bar_width + 5, y + bar_height - 2), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1);
    }
}

std::string MultimodalEmotionAnalyzer::formatEmotionText(const EmotionResult& result) {
    if (result.dominant_emotion == EmotionType::UNKNOWN) {
        return "Unknown";
    }
    
    std::stringstream ss;
    ss << emotionToString(result.dominant_emotion) 
       << " (" << std::fixed << std::setprecision(1) << (result.confidence * 100) << "%)";
    
    return ss.str();
}

} // namespace EmotionAnalysis
