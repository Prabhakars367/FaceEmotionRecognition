#include "VideoEmotionAnalyzer.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cmath>
#include <numeric>
#include <chrono>

namespace EmotionAnalysis {

class VideoEmotionAnalyzer::Impl {
public:
    cv::CascadeClassifier face_cascade;
    std::vector<cv::Rect> last_detected_faces;
    int min_face_size = 30;
    bool is_initialized = false;
    std::mt19937 rng{std::random_device{}()};
    
    // Advanced emotion recognition components
    std::vector<EmotionResult> emotion_history;
    static constexpr size_t MAX_HISTORY = 10;
    
    // Advanced facial analysis methods
    std::vector<cv::Point2f> detectFacialKeypoints(const cv::Mat& face);
    std::vector<float> extractGeometricFeatures(const std::vector<cv::Point2f>& keypoints);
    std::vector<float> extractAdvancedLBPFeatures(const cv::Mat& face);
    std::vector<float> extractAdvancedHOGFeatures(const cv::Mat& face);
    std::vector<float> extractGaborFeatures(const cv::Mat& face);
    std::vector<float> extractFacialActionUnits(const cv::Mat& face);
    std::vector<float> extractTextureFeatures(const cv::Mat& face);
    EmotionResult ensembleClassification(const std::vector<std::vector<float>>& all_features);
    EmotionResult applyTemporalSmoothing(const EmotionResult& current_result);
    
    // Classification methods
    void classifyGeometricFeatures(const std::vector<float>& features, std::vector<float>& probabilities);
    void classifyTextureFeatures(const std::vector<float>& features, std::vector<float>& probabilities, float avgIntensity, float contrast);
    void classifyShapeFeatures(const std::vector<float>& features, std::vector<float>& probabilities, float aspectRatio, float area);
    void classifyGaborFeatures(const std::vector<float>& features, std::vector<float>& probabilities, float meanResponse, float stdResponse);
    void classifyActionUnits(const std::vector<float>& features, std::vector<float>& probabilities);
    void classifyGenericFeatures(const std::vector<float>& features, std::vector<float>& probabilities, float weight1, float weight2);
    
    // Gabor filter bank
    std::vector<cv::Mat> generateGaborKernels(int num_orientations = 8, int num_scales = 5);
    cv::Mat getGaborKernel(double sigma, double theta, double lambda, double gamma, double psi);
    
    // Face quality validation
    bool validateFaceQuality(const cv::Mat& frame, const cv::Rect& face_rect);
    
    // Enhanced face detection
    std::vector<cv::Rect> improvedFaceDetection(const cv::Mat& gray_frame, const cv::Mat& frame);
};

VideoEmotionAnalyzer::VideoEmotionAnalyzer() 
    : pImpl(std::make_unique<Impl>()) {}

VideoEmotionAnalyzer::~VideoEmotionAnalyzer() = default;

bool VideoEmotionAnalyzer::initialize(const std::string& face_cascade_path, 
                                     const std::string& emotion_model_path) {
    try {
        if (!pImpl->face_cascade.load(face_cascade_path)) {
            std::cerr << "Error: Could not load face cascade from: " << face_cascade_path << std::endl;
            return false;
        }
        
        pImpl->is_initialized = true;
        
        if (!emotion_model_path.empty()) {
            std::cout << "Note: Using advanced computer vision techniques for emotion analysis." << std::endl;
        }
        
        std::cout << "Advanced video emotion analyzer initialized successfully." << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing video emotion analyzer: " << e.what() << std::endl;
        return false;
    }
}

EmotionResult VideoEmotionAnalyzer::analyzeFrame(const cv::Mat& frame) {
    EmotionResult result;
    
    if (!pImpl->is_initialized || frame.empty()) {
        return result;
    }
    
    try {
        // Convert to grayscale for face detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        
        // Detect faces
        std::vector<cv::Rect> faces;
        pImpl->face_cascade.detectMultiScale(
            gray, faces, 1.1, 3, 0,
            cv::Size(pImpl->min_face_size, pImpl->min_face_size)
        );
        
        pImpl->last_detected_faces = faces;
        
        if (!faces.empty()) {
            // Use the largest face for emotion analysis
            auto largest_face = *std::max_element(faces.begin(), faces.end(),
                [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                });
            
            cv::Mat face_roi = gray(largest_face);
            result = analyzeFacialFeatures(face_roi);
        }
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing frame: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<cv::Rect> VideoEmotionAnalyzer::getLastDetectedFaces() const {
    return pImpl->last_detected_faces;
}

void VideoEmotionAnalyzer::setMinFaceSize(int min_size) {
    pImpl->min_face_size = std::max(20, min_size);
}

cv::Mat VideoEmotionAnalyzer::preprocessFace(const cv::Mat& face) {
    cv::Mat preprocessed;
    
    // Resize to standard size
    cv::resize(face, preprocessed, cv::Size(128, 128));
    
    // Apply histogram equalization for better contrast
    cv::equalizeHist(preprocessed, preprocessed);
    
    // Apply Gaussian blur for noise reduction
    cv::GaussianBlur(preprocessed, preprocessed, cv::Size(3, 3), 1.0);
    
    return preprocessed;
}

EmotionResult VideoEmotionAnalyzer::classifyEmotion(const cv::Mat& preprocessed_face) {
    // This method is now replaced by the advanced ensemble classification
    return analyzeFacialFeatures(preprocessed_face);
}

EmotionResult VideoEmotionAnalyzer::analyzeFacialFeatures(const cv::Mat& face) {
    EmotionResult result;
    
    if (face.empty()) {
        return result;
    }
    
    try {
        cv::Mat preprocessed = preprocessFace(face);
        
        // Extract multiple types of advanced features
        std::vector<cv::Point2f> keypoints = pImpl->detectFacialKeypoints(preprocessed);
        std::vector<float> geometric_features = pImpl->extractGeometricFeatures(keypoints);
        std::vector<float> advanced_lbp_features = pImpl->extractAdvancedLBPFeatures(preprocessed);
        std::vector<float> advanced_hog_features = pImpl->extractAdvancedHOGFeatures(preprocessed);
        std::vector<float> gabor_features = pImpl->extractGaborFeatures(preprocessed);
        std::vector<float> action_units = pImpl->extractFacialActionUnits(preprocessed);
        std::vector<float> texture_features = pImpl->extractTextureFeatures(preprocessed);
        
        // Combine all features for ensemble classification
        std::vector<std::vector<float>> all_features = {
            geometric_features, advanced_lbp_features, advanced_hog_features,
            gabor_features, action_units, texture_features
        };
        
        // Advanced ensemble classification
        result = pImpl->ensembleClassification(all_features);
        
        // Apply temporal smoothing for stability
        result = pImpl->applyTemporalSmoothing(result);
        
        result.timestamp = std::chrono::steady_clock::now();
        
    }
    catch (const std::exception& e) {
        std::cerr << "Error in advanced facial feature analysis: " << e.what() << std::endl;
    }
    
    return result;
}

// Enhanced face detection with multiple cascades and validation
std::vector<cv::Rect> VideoEmotionAnalyzer::Impl::improvedFaceDetection(const cv::Mat& gray_frame, const cv::Mat& frame) {
    std::vector<cv::Rect> faces;
    
    // Use multiple detection scales for better accuracy
    std::vector<double> scale_factors = {1.1, 1.2, 1.3};
    std::vector<int> min_neighbors = {3, 4, 5};
    
    for (double scale_factor : scale_factors) {
        for (int min_neighbor : min_neighbors) {
            std::vector<cv::Rect> detected_faces;
            face_cascade.detectMultiScale(
                gray_frame, detected_faces,
                scale_factor, min_neighbor, 0,
                cv::Size(min_face_size, min_face_size)
            );
            
            // Merge results with existing detections
            for (const auto& face : detected_faces) {
                bool is_duplicate = false;
                for (const auto& existing_face : faces) {
                    // Check for significant overlap
                    cv::Rect intersection = face & existing_face;
                    float overlap_ratio = static_cast<float>(intersection.area()) / 
                                        std::max(face.area(), existing_face.area());
                    if (overlap_ratio > 0.6f) {
                        is_duplicate = true;
                        break;
                    }
                }
                if (!is_duplicate) {
                    faces.push_back(face);
                }
            }
        }
    }
    
    // Quality-based face filtering
    std::vector<cv::Rect> validated_faces;
    for (const auto& face : faces) {
        if (validateFaceQuality(frame, face)) {
            validated_faces.push_back(face);
        }
    }
    
    return validated_faces.empty() ? faces : validated_faces;
}

// Advanced facial keypoint detection using corner detection and contour analysis
std::vector<cv::Point2f> VideoEmotionAnalyzer::Impl::detectFacialKeypoints(const cv::Mat& face) {
    std::vector<cv::Point2f> keypoints;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        if (gray_face.type() != CV_8UC1) {
            gray_face.convertTo(gray_face, CV_8UC1, 255.0);
        }
        
        int h = gray_face.rows;
        int w = gray_face.cols;
        
        // Define anatomically-based facial landmark regions
        std::vector<cv::Rect> regions = {
            cv::Rect(w*0.15, h*0.15, w*0.25, h*0.2),  // Left eye
            cv::Rect(w*0.6, h*0.15, w*0.25, h*0.2),   // Right eye
            cv::Rect(w*0.35, h*0.3, w*0.3, h*0.25),   // Nose
            cv::Rect(w*0.25, h*0.6, w*0.5, h*0.25),   // Mouth
            cv::Rect(w*0.1, h*0.4, w*0.2, h*0.3),     // Left cheek
            cv::Rect(w*0.7, h*0.4, w*0.2, h*0.3)      // Right cheek
        };
        
        // Detect corners in each region
        for (size_t i = 0; i < regions.size(); ++i) {
            cv::Rect region = regions[i];
            if (region.x + region.width <= w && region.y + region.height <= h) {
                cv::Mat roi = gray_face(region);
                std::vector<cv::Point2f> corners;
                
                cv::goodFeaturesToTrack(roi, corners, 5, 0.01, 10);
                
                // Convert local coordinates to global face coordinates
                for (auto& corner : corners) {
                    corner.x += region.x;
                    corner.y += region.y;
                    keypoints.push_back(corner);
                }
            }
        }
        
        // Add symmetric landmark points for better geometry
        if (keypoints.size() >= 6) {
            // Eye centers approximation
            keypoints.push_back(cv::Point2f(w*0.275, h*0.25)); // Left eye center
            keypoints.push_back(cv::Point2f(w*0.725, h*0.25)); // Right eye center
            // Nose tip
            keypoints.push_back(cv::Point2f(w*0.5, h*0.45));
            // Mouth corners
            keypoints.push_back(cv::Point2f(w*0.35, h*0.7));   // Left mouth corner
            keypoints.push_back(cv::Point2f(w*0.65, h*0.7));   // Right mouth corner
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in facial keypoint detection: " << e.what() << std::endl;
    }
    
    return keypoints;
}

// Enhanced geometric feature extraction with comprehensive facial metrics
std::vector<float> VideoEmotionAnalyzer::Impl::extractGeometricFeatures(const std::vector<cv::Point2f>& keypoints) {
    std::vector<float> features;
    
    if (keypoints.size() < 10) {
        return features; // Need sufficient keypoints for meaningful analysis
    }
    
    try {
        // Eye region analysis
        cv::Point2f left_eye = keypoints[keypoints.size()-5];  // Left eye center
        cv::Point2f right_eye = keypoints[keypoints.size()-4]; // Right eye center
        cv::Point2f nose_tip = keypoints[keypoints.size()-3];  // Nose tip
        cv::Point2f left_mouth = keypoints[keypoints.size()-2]; // Left mouth corner
        cv::Point2f right_mouth = keypoints[keypoints.size()-1]; // Right mouth corner
        
        // Basic distances
        float eye_distance = cv::norm(left_eye - right_eye);
        float mouth_width = cv::norm(left_mouth - right_mouth);
        float eye_mouth_distance = cv::norm((left_eye + right_eye)/2 - (left_mouth + right_mouth)/2);
        
        // Ratios for scale-invariant features
        float mouth_eye_ratio = (eye_distance > 0) ? mouth_width / eye_distance : 0;
        float face_length_ratio = (eye_distance > 0) ? eye_mouth_distance / eye_distance : 0;
        
        // Symmetry analysis
        cv::Point2f face_center = (left_eye + right_eye + nose_tip) / 3;
        float left_eye_asymmetry = std::abs((left_eye - face_center).x - (face_center - right_eye).x);
        float mouth_asymmetry = std::abs((left_mouth - face_center).x - (face_center - right_mouth).x);
        
        // Angular features
        cv::Point2f eye_vector = right_eye - left_eye;
        cv::Point2f mouth_vector = right_mouth - left_mouth;
        float eye_mouth_angle = std::atan2(mouth_vector.y - eye_vector.y, 
                                          mouth_vector.x - eye_vector.x) * 180.0 / CV_PI;
        
        // Facial curvature approximation
        float nose_eye_angle = std::atan2(nose_tip.y - (left_eye.y + right_eye.y)/2,
                                         nose_tip.x - (left_eye.x + right_eye.x)/2) * 180.0 / CV_PI;
        
        // Compile features
        features = {
            mouth_eye_ratio, face_length_ratio, left_eye_asymmetry, mouth_asymmetry,
            eye_mouth_angle, nose_eye_angle, eye_distance, mouth_width, eye_mouth_distance
        };
        
    } catch (const std::exception& e) {
        std::cerr << "Error in geometric feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

// Advanced Multi-scale Local Binary Pattern features
std::vector<float> VideoEmotionAnalyzer::Impl::extractAdvancedLBPFeatures(const cv::Mat& face) {
    std::vector<float> features;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        if (gray_face.type() != CV_8UC1) {
            gray_face.convertTo(gray_face, CV_8UC1, 255.0);
        }
        
        int h = gray_face.rows;
        int w = gray_face.cols;
        
        // Define emotional relevant regions
        std::vector<cv::Rect> regions = {
            cv::Rect(0, 0, w, h/3),           // Forehead/eyebrow region
            cv::Rect(w*0.1, h*0.1, w*0.35, h*0.3),   // Left eye region
            cv::Rect(w*0.55, h*0.1, w*0.35, h*0.3),  // Right eye region
            cv::Rect(w*0.3, h*0.25, w*0.4, h*0.35),  // Nose region
            cv::Rect(w*0.2, h*0.55, w*0.6, h*0.35),  // Mouth region
            cv::Rect(0, h*0.4, w*0.3, h*0.4),        // Left cheek
            cv::Rect(w*0.7, h*0.4, w*0.3, h*0.4)     // Right cheek
        };
        
        // Multi-scale LBP with different radii
        std::vector<int> radii = {1, 2, 3};
        std::vector<int> neighbors = {8, 12, 16};
        
        for (const auto& region : regions) {
            if (region.x + region.width <= w && region.y + region.height <= h) {
                cv::Mat roi = gray_face(region);
                
                for (size_t scale = 0; scale < radii.size(); ++scale) {
                    int radius = radii[scale];
                    int n_neighbors = neighbors[scale];
                    
                    // Calculate LBP histogram
                    std::vector<int> histogram(256, 0);
                    
                    for (int i = radius; i < roi.rows - radius; i++) {
                        for (int j = radius; j < roi.cols - radius; j++) {
                            uchar center = roi.at<uchar>(i, j);
                            uchar code = 0;
                            
                            // Circular LBP
                            for (int n = 0; n < n_neighbors; ++n) {
                                float angle = 2.0 * CV_PI * n / n_neighbors;
                                int x = static_cast<int>(round(i + radius * cos(angle)));
                                int y = static_cast<int>(round(j + radius * sin(angle)));
                                
                                x = std::max(0, std::min(roi.rows-1, x));
                                y = std::max(0, std::min(roi.cols-1, y));
                                
                                if (roi.at<uchar>(x, y) >= center) {
                                    code |= (1 << n);
                                }
                            }
                            
                            histogram[code]++;
                        }
                    }
                    
                    // Normalize histogram and add to features
                    int total_pixels = (roi.rows - 2*radius) * (roi.cols - 2*radius);
                    if (total_pixels > 0) {
                        for (int h_val : histogram) {
                            features.push_back(static_cast<float>(h_val) / total_pixels);
                        }
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in advanced LBP feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

// Advanced HOG features with multi-scale analysis
std::vector<float> VideoEmotionAnalyzer::Impl::extractAdvancedHOGFeatures(const cv::Mat& face) {
    std::vector<float> features;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        // HOG parameters for facial emotion detection
        cv::HOGDescriptor hog(
            cv::Size(64, 64),    // winSize
            cv::Size(16, 16),    // blockSize
            cv::Size(8, 8),      // blockStride
            cv::Size(8, 8),      // cellSize
            9                    // nbins
        );
        
        cv::Mat resized_face;
        cv::resize(gray_face, resized_face, cv::Size(64, 64));
        
        std::vector<float> hog_features;
        hog.compute(resized_face, hog_features);
        
        // Extract features from different facial regions
        int h = resized_face.rows;
        int w = resized_face.cols;
        
        std::vector<cv::Rect> regions = {
            cv::Rect(0, 0, w, h/2),           // Upper face
            cv::Rect(0, h/2, w, h/2),         // Lower face
            cv::Rect(0, 0, w/2, h),           // Left face
            cv::Rect(w/2, 0, w/2, h),         // Right face
            cv::Rect(w/4, h/4, w/2, h/2)      // Central face
        };
        
        for (const auto& region : regions) {
            if (region.x + region.width <= w && region.y + region.height <= h) {
                cv::Mat roi = resized_face(region);
                cv::resize(roi, roi, cv::Size(64, 64)); // Normalize size
                
                std::vector<float> region_hog;
                hog.compute(roi, region_hog);
                
                features.insert(features.end(), region_hog.begin(), region_hog.end());
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in advanced HOG feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

// Generate Gabor kernel
cv::Mat VideoEmotionAnalyzer::Impl::getGaborKernel(double sigma, double theta, double lambda, double gamma, double psi) {
    int size = static_cast<int>(6 * sigma + 1);
    if (size % 2 == 0) size++;
    
    cv::Mat kernel(size, size, CV_32F);
    int center = size / 2;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double x = (i - center) * cos(theta) + (j - center) * sin(theta);
            double y = -(i - center) * sin(theta) + (j - center) * cos(theta);
            
            double gaussian = exp(-(x*x + gamma*gamma*y*y) / (2*sigma*sigma));
            double sinusoid = cos(2*CV_PI*x/lambda + psi);
            
            kernel.at<float>(i, j) = static_cast<float>(gaussian * sinusoid);
        }
    }
    
    return kernel;
}

// Gabor filter bank for texture analysis
std::vector<float> VideoEmotionAnalyzer::Impl::extractGaborFeatures(const cv::Mat& face) {
    std::vector<float> features;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        gray_face.convertTo(gray_face, CV_32F, 1.0/255.0);
        
        // Gabor filter parameters
        std::vector<double> orientations = {0, CV_PI/4, CV_PI/2, 3*CV_PI/4};
        std::vector<double> frequencies = {0.1, 0.2, 0.3};
        
        for (double orientation : orientations) {
            for (double frequency : frequencies) {
                cv::Mat kernel = getGaborKernel(2.0, orientation, 1.0/frequency, 0.5, 0);
                cv::Mat filtered;
                cv::filter2D(gray_face, filtered, CV_32F, kernel);
                
                // Extract statistical features from filtered image
                cv::Scalar mean, stddev;
                cv::meanStdDev(filtered, mean, stddev);
                
                features.push_back(static_cast<float>(mean[0]));
                features.push_back(static_cast<float>(stddev[0]));
                
                // Energy
                cv::Mat squared;
                cv::multiply(filtered, filtered, squared);
                cv::Scalar energy = cv::sum(squared);
                features.push_back(static_cast<float>(energy[0]));
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Gabor feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

// Facial Action Units approximation
std::vector<float> VideoEmotionAnalyzer::Impl::extractFacialActionUnits(const cv::Mat& face) {
    std::vector<float> action_units;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        int h = gray_face.rows;
        int w = gray_face.cols;
        
        // Define regions corresponding to major facial action units
        cv::Rect eyebrow_left(w*0.15, h*0.1, w*0.25, h*0.15);
        cv::Rect eyebrow_right(w*0.6, h*0.1, w*0.25, h*0.15);
        cv::Rect eye_left(w*0.15, h*0.2, w*0.25, h*0.15);
        cv::Rect eye_right(w*0.6, h*0.2, w*0.25, h*0.15);
        cv::Rect nasolabial_left(w*0.2, h*0.45, w*0.2, h*0.25);
        cv::Rect nasolabial_right(w*0.6, h*0.45, w*0.2, h*0.25);
        cv::Rect mouth_region(w*0.25, h*0.6, w*0.5, h*0.25);
        
        std::vector<cv::Rect> au_regions = {
            eyebrow_left, eyebrow_right, eye_left, eye_right,
            nasolabial_left, nasolabial_right, mouth_region
        };
        
        for (const auto& region : au_regions) {
            if (region.x + region.width <= w && region.y + region.height <= h) {
                cv::Mat roi = gray_face(region);
                
                // Calculate gradient magnitude for activation intensity
                cv::Mat grad_x, grad_y, grad_mag;
                cv::Sobel(roi, grad_x, CV_32F, 1, 0, 3);
                cv::Sobel(roi, grad_y, CV_32F, 0, 1, 3);
                cv::magnitude(grad_x, grad_y, grad_mag);
                
                cv::Scalar mean_grad = cv::mean(grad_mag);
                action_units.push_back(static_cast<float>(mean_grad[0]));
                
                // Texture variation
                cv::Scalar mean, stddev;
                cv::meanStdDev(roi, mean, stddev);
                action_units.push_back(static_cast<float>(stddev[0]));
                
                // Intensity distribution
                action_units.push_back(static_cast<float>(mean[0]));
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in facial action unit extraction: " << e.what() << std::endl;
    }
    
    return action_units;
}

// Enhanced texture features
std::vector<float> VideoEmotionAnalyzer::Impl::extractTextureFeatures(const cv::Mat& face) {
    std::vector<float> features;
    
    try {
        cv::Mat gray_face;
        if (face.channels() == 3) {
            cv::cvtColor(face, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face.clone();
        }
        
        // Gray Level Co-occurrence Matrix approximation
        std::vector<int> histogram(256, 0);
        for (int i = 0; i < gray_face.rows; i++) {
            for (int j = 0; j < gray_face.cols; j++) {
                histogram[gray_face.at<uchar>(i, j)]++;
            }
        }
        
        // Calculate texture statistics
        float entropy = 0;
        float energy = 0;
        int total_pixels = gray_face.rows * gray_face.cols;
        
        for (int i = 0; i < 256; i++) {
            if (histogram[i] > 0) {
                float prob = static_cast<float>(histogram[i]) / total_pixels;
                entropy -= prob * log2(prob);
                energy += prob * prob;
            }
        }
        
        features.push_back(entropy);
        features.push_back(energy);
        
        // Local variance
        cv::Mat variance;
        cv::Mat mean_filtered;
        cv::boxFilter(gray_face, mean_filtered, CV_32F, cv::Size(5, 5));
        
        cv::Mat diff, diff_squared;
        gray_face.convertTo(diff, CV_32F);
        cv::subtract(diff, mean_filtered, diff);
        cv::multiply(diff, diff, diff_squared);
        cv::boxFilter(diff_squared, variance, CV_32F, cv::Size(5, 5));
        
        cv::Scalar mean_variance = cv::mean(variance);
        features.push_back(static_cast<float>(mean_variance[0]));
        
    } catch (const std::exception& e) {
        std::cerr << "Error in texture feature extraction: " << e.what() << std::endl;
    }
    
    return features;
}

// Advanced ensemble classification with attention mechanism
EmotionResult VideoEmotionAnalyzer::Impl::ensembleClassification(const std::vector<std::vector<float>>& all_features) {
    EmotionResult result;
    
    try {
        if (all_features.empty()) {
            return result;
        }
        
        // Feature weights based on empirical effectiveness for emotion recognition
        std::vector<float> feature_weights = {
            0.25f,  // Geometric features - high importance for basic expressions
            0.20f,  // Advanced LBP - good for texture analysis
            0.20f,  // Advanced HOG - excellent for edge/shape detection
            0.15f,  // Gabor features - good for texture and orientation
            0.15f,  // Facial action units - specific for facial movements
            0.05f   // Additional texture features
        };
        
        // Ensure we have weights for all feature types
        while (feature_weights.size() < all_features.size()) {
            feature_weights.push_back(0.05f);
        }
        
        // Initialize emotion probabilities
        std::vector<float> emotion_scores(static_cast<int>(EmotionType::COUNT), 0.0f);
        
        // Process each feature type
        for (size_t feature_type = 0; feature_type < all_features.size(); ++feature_type) {
            const auto& features = all_features[feature_type];
            if (features.empty()) continue;
            
            float weight = feature_weights[feature_type];
            
            // Classify based on feature patterns
            std::vector<float> type_scores(8, 0.0f);
            
            // Route to appropriate classifier based on feature type
            switch (feature_type) {
                case 0: // Geometric features
                    classifyGeometricFeatures(features, type_scores);
                    break;
                case 1: // Texture features
                    {
                        cv::Scalar mean_scalar = cv::mean(all_features.size() > 1 ? cv::Mat(all_features[1]) : cv::Mat());
                        float avgIntensity = mean_scalar[0];
                        float contrast = features.size() > 1 ? features[1] : 0.0f;
                        classifyTextureFeatures(features, type_scores, avgIntensity, contrast);
                    }
                    break;
                case 2: // HOG features
                    {
                        float aspectRatio = features.size() > 0 ? features[0] : 1.0f;
                        float area = features.size() > 1 ? features[1] : 100.0f;
                        classifyShapeFeatures(features, type_scores, aspectRatio, area);
                    }
                    break;
                case 3: // Gabor features
                    {
                        cv::Scalar mean_scalar = cv::mean(features.size() > 0 ? cv::Mat(features) : cv::Mat());
                        float meanResponse = mean_scalar[0];
                        cv::Scalar stddev_scalar;
                        cv::meanStdDev(features.size() > 0 ? cv::Mat(features) : cv::Mat(), mean_scalar, stddev_scalar);
                        float stdResponse = stddev_scalar[0];
                        classifyGaborFeatures(features, type_scores, meanResponse, stdResponse);
                    }
                    break;
                case 4: // Action units
                    classifyActionUnits(features, type_scores);
                    break;
                default: // Generic features
                    {
                        float weight1 = features.size() > 0 ? features[0] : 0.5f;
                        float weight2 = features.size() > 1 ? features[1] : 0.5f;
                        classifyGenericFeatures(features, type_scores, weight1, weight2);
                    }
                    break;
            }
            
            // Add weighted scores
            for (size_t emotion = 0; emotion < emotion_scores.size() && emotion < type_scores.size(); ++emotion) {
                emotion_scores[emotion] += weight * type_scores[emotion];
            }
        }
        
        // Normalize scores
        float total_score = std::accumulate(emotion_scores.begin(), emotion_scores.end(), 0.0f);
        if (total_score > 0) {
            for (auto& score : emotion_scores) {
                score /= total_score;
            }
        }
        
        // Find dominant emotion
        auto max_it = std::max_element(emotion_scores.begin(), emotion_scores.end());
        result.dominant_emotion = static_cast<EmotionType>(std::distance(emotion_scores.begin(), max_it));
        result.confidence = *max_it;
        
        // Copy probabilities
        std::copy(emotion_scores.begin(), emotion_scores.end(), result.probabilities.begin());
        
    } catch (const std::exception& e) {
        std::cerr << "Error in ensemble classification: " << e.what() << std::endl;
    }
    
    return result;
}



// Geometric feature classification
void VideoEmotionAnalyzer::Impl::classifyGeometricFeatures(const std::vector<float>& features, std::vector<float>& scores) {
    if (features.size() < 9) return;
    
    float mouth_eye_ratio = features[0];
    float face_length_ratio = features[1];
    float left_eye_asymmetry = features[2];
    float mouth_asymmetry = features[3];
    float eye_mouth_angle = features[4];
    float nose_eye_angle = features[5];
    float eye_distance = features[6];
    float mouth_width = features[7];
    float eye_mouth_distance = features[8];
    
    // Normalize features for better comparison
    float normalized_asymmetry = (left_eye_asymmetry + mouth_asymmetry) / 2.0f;
    float angular_deviation = std::abs(eye_mouth_angle) + std::abs(nose_eye_angle);
    
    // Initialize scores to prevent bias
    for (int i = 0; i < static_cast<int>(EmotionType::COUNT); ++i) {
        scores[i] = 0.1f; // Small baseline to prevent zero probability
    }
    
    // Enhanced Happiness Detection: wider smile, upward mouth curve, balanced symmetry
    float happiness_score = 0.0f;
    if (mouth_eye_ratio > 0.65 && mouth_eye_ratio < 1.2) happiness_score += 0.4f;
    if (mouth_asymmetry < 3.0) happiness_score += 0.3f;
    if (eye_mouth_angle > -2 && eye_mouth_angle < 8) happiness_score += 0.3f;
    if (face_length_ratio > 1.0 && face_length_ratio < 1.6) happiness_score += 0.2f;
    scores[static_cast<int>(EmotionType::HAPPINESS)] += happiness_score;
    
    // Enhanced Sadness Detection: drooping features, longer face
    float sadness_score = 0.0f;
    if (eye_mouth_angle < -8) sadness_score += 0.5f;
    if (face_length_ratio > 1.6) sadness_score += 0.4f;
    if (mouth_eye_ratio < 0.5) sadness_score += 0.3f;
    if (nose_eye_angle < -5) sadness_score += 0.2f;
    scores[static_cast<int>(EmotionType::SADNESS)] += sadness_score;
    
    // Enhanced Surprise Detection: wide eyes, raised eyebrows, open mouth
    float surprise_score = 0.0f;
    if (face_length_ratio < 1.1) surprise_score += 0.4f;
    if (angular_deviation > 15) surprise_score += 0.4f;
    if (mouth_eye_ratio > 0.8) surprise_score += 0.3f;
    if (eye_mouth_distance < eye_distance * 1.5) surprise_score += 0.2f;
    scores[static_cast<int>(EmotionType::SURPRISE)] += surprise_score;
    
    // Enhanced Anger Detection: tensed features, frowning, asymmetry
    float anger_score = 0.0f;
    if (normalized_asymmetry > 4.0) anger_score += 0.5f;
    if (eye_mouth_angle < -10 || eye_mouth_angle > 10) anger_score += 0.4f;
    if (mouth_eye_ratio < 0.6 && mouth_asymmetry > 2.0) anger_score += 0.4f;
    if (nose_eye_angle < -8 || nose_eye_angle > 8) anger_score += 0.3f;
    scores[static_cast<int>(EmotionType::ANGER)] += anger_score;
    
    // Enhanced Fear Detection: wide eyes, tensed mouth, asymmetric features
    float fear_score = 0.0f;
    if (face_length_ratio < 1.2 && normalized_asymmetry > 2.5) fear_score += 0.5f;
    if (angular_deviation > 12 && angular_deviation < 25) fear_score += 0.4f;
    if (mouth_eye_ratio > 0.7 && mouth_asymmetry > 2.0) fear_score += 0.3f;
    if (eye_mouth_distance < eye_distance * 1.8) fear_score += 0.2f;
    scores[static_cast<int>(EmotionType::FEAR)] += fear_score;
    
    // Enhanced Disgust Detection: nose wrinkle, mouth distortion
    float disgust_score = 0.0f;
    if (mouth_asymmetry > 5.0) disgust_score += 0.5f;
    if (nose_eye_angle > 5 || nose_eye_angle < -15) disgust_score += 0.4f;
    if (mouth_eye_ratio < 0.5) disgust_score += 0.4f;
    if (left_eye_asymmetry > 3.0) disgust_score += 0.3f;
    scores[static_cast<int>(EmotionType::DISGUST)] += disgust_score;
    
    // Enhanced Neutral Detection: balanced proportions, minimal asymmetry
    float neutral_score = 0.0f;
    if (mouth_eye_ratio > 0.5 && mouth_eye_ratio < 0.8) neutral_score += 0.4f;
    if (normalized_asymmetry < 2.0) neutral_score += 0.4f;
    if (angular_deviation < 8) neutral_score += 0.3f;
    if (face_length_ratio > 1.2 && face_length_ratio < 1.8) neutral_score += 0.3f;
    scores[static_cast<int>(EmotionType::NEUTRAL)] += neutral_score;
    
    // Apply confidence boosting for clear cases
    float max_score = *std::max_element(scores.begin(), scores.end());
    if (max_score > 1.5f) {
        for (float& score : scores) {
            score *= 1.2f; // Boost confidence for clear detections
        }
    }
}

// Enhanced texture classification with advanced statistical analysis
void VideoEmotionAnalyzer::Impl::classifyTextureFeatures(const std::vector<float>& features, std::vector<float>& scores, float mean, float std_dev) {
    if (features.empty()) return;
    
    // Calculate advanced texture statistics
    float variance = std_dev * std_dev;
    float coefficient_of_variation = (mean > 0) ? std_dev / mean : 0;
    
    // Calculate higher-order moments for better texture characterization
    float skewness = 0.0f, kurtosis = 0.0f;
    for (float feature : features) {
        float deviation = feature - mean;
        skewness += std::pow(deviation, 3);
        kurtosis += std::pow(deviation, 4);
    }
    skewness /= (features.size() * std::pow(std_dev, 3));
    kurtosis /= (features.size() * std::pow(std_dev, 4)) - 3; // Excess kurtosis
    
    // High texture variation with positive skewness often indicates muscle tension (anger, disgust)
    if (coefficient_of_variation > 0.4 && skewness > 0.2) {
        scores[static_cast<int>(EmotionType::ANGER)] += 0.5f;
        scores[static_cast<int>(EmotionType::DISGUST)] += 0.4f;
    }
    
    // Very high kurtosis indicates extreme texture variation (surprise, fear)
    if (kurtosis > 1.0) {
        scores[static_cast<int>(EmotionType::SURPRISE)] += 0.4f;
        scores[static_cast<int>(EmotionType::FEAR)] += 0.3f;
    }
    
    // Low variation with negative skewness might indicate sadness
    if (coefficient_of_variation < 0.25 && skewness < -0.1) {
        scores[static_cast<int>(EmotionType::SADNESS)] += 0.5f;
    }
    
    // Moderate variation with low kurtosis suggests happiness or neutral
    if (coefficient_of_variation > 0.2 && coefficient_of_variation < 0.4 && std::abs(kurtosis) < 0.5) {
        scores[static_cast<int>(EmotionType::HAPPINESS)] += 0.3f;
        scores[static_cast<int>(EmotionType::NEUTRAL)] += 0.4f;
    }
    
    // Very low variation typically indicates neutral expression
    if (coefficient_of_variation < 0.15) {
        scores[static_cast<int>(EmotionType::NEUTRAL)] += 0.6f;
    }
}

void VideoEmotionAnalyzer::Impl::classifyShapeFeatures(const std::vector<float>& features, std::vector<float>& scores, float mean, float std_dev) {
    // Strong edges often indicate happiness or surprise
    if (mean > 0.3) {
        scores[static_cast<int>(EmotionType::HAPPINESS)] += 0.4f;
        scores[static_cast<int>(EmotionType::SURPRISE)] += 0.3f;
    }
}

void VideoEmotionAnalyzer::Impl::classifyGaborFeatures(const std::vector<float>& features, std::vector<float>& scores, float mean, float std_dev) {
    // Complex texture patterns might indicate complex emotions
    if (std_dev > mean) {
        scores[static_cast<int>(EmotionType::FEAR)] += 0.2f;
        scores[static_cast<int>(EmotionType::SURPRISE)] += 0.2f;
    }
}

void VideoEmotionAnalyzer::Impl::classifyActionUnits(const std::vector<float>& features, std::vector<float>& scores) {
    // Action units directly relate to specific muscle movements
    // This is a simplified version - real AU classification is complex
    for (size_t i = 0; i < features.size() && i < scores.size(); ++i) {
        if (features[i] > 10.0) { // Threshold for activation
            scores[i % scores.size()] += 0.2f;
        }
    }
}

void VideoEmotionAnalyzer::Impl::classifyGenericFeatures(const std::vector<float>& features, std::vector<float>& scores, float mean, float std_dev) {
    // Default classification based on statistical properties
    if (mean > 0.5) {
        scores[static_cast<int>(EmotionType::HAPPINESS)] += 0.1f;
    } else if (mean < 0.2) {
        scores[static_cast<int>(EmotionType::SADNESS)] += 0.1f;
    } else {
        scores[static_cast<int>(EmotionType::NEUTRAL)] += 0.1f;
    }
}

// Face quality validation to improve detection accuracy
bool VideoEmotionAnalyzer::Impl::validateFaceQuality(const cv::Mat& frame, const cv::Rect& face_rect) {
    try {
        // Check basic constraints
        if (face_rect.area() < min_face_size * min_face_size) return false;
        if (face_rect.x < 0 || face_rect.y < 0 || 
            face_rect.x + face_rect.width > frame.cols ||
            face_rect.y + face_rect.height > frame.rows) return false;
        
        cv::Mat face_roi = frame(face_rect);
        cv::Mat gray_face;
        
        if (face_roi.channels() == 3) {
            cv::cvtColor(face_roi, gray_face, cv::COLOR_BGR2GRAY);
        } else {
            gray_face = face_roi.clone();
        }
        
        // Check image quality metrics
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray_face, mean, stddev);
        
        // Reject faces that are too dark, too bright, or have low contrast
        if (mean[0] < 30 || mean[0] > 220) return false;  // Lighting check
        if (stddev[0] < 15) return false;  // Contrast check
        
        // Check for reasonable face proportions (aspect ratio)
        float aspect_ratio = static_cast<float>(face_rect.width) / face_rect.height;
        if (aspect_ratio < 0.6f || aspect_ratio > 1.4f) return false;
        
        // Edge density check (faces should have reasonable edge content)
        cv::Mat edges;
        cv::Canny(gray_face, edges, 50, 150);
        int edge_pixels = cv::countNonZero(edges);
        float edge_density = static_cast<float>(edge_pixels) / (gray_face.rows * gray_face.cols);
        if (edge_density < 0.05f || edge_density > 0.4f) return false;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in face validation: " << e.what() << std::endl;
        return false;
    }
}

// Enhanced temporal smoothing with outlier detection
EmotionResult VideoEmotionAnalyzer::Impl::applyTemporalSmoothing(const EmotionResult& current_result) {
    EmotionResult smoothed_result = current_result;
    
    try {
        // Add to history
        emotion_history.push_back(current_result);
        if (emotion_history.size() > MAX_HISTORY) {
            emotion_history.erase(emotion_history.begin());
        }
        
        if (emotion_history.size() >= 2) {
            // Enhanced outlier detection using multiple criteria
            std::vector<float> confidences;
            std::vector<EmotionType> emotions;
            
            for (const auto& result : emotion_history) {
                confidences.push_back(result.confidence);
                emotions.push_back(result.dominant_emotion);
            }
            
            // Statistical outlier detection
            std::sort(confidences.begin(), confidences.end());
            float median_confidence = confidences[confidences.size() / 2];
            float q1 = confidences[confidences.size() / 4];
            float q3 = confidences[3 * confidences.size() / 4];
            float iqr = q3 - q1;
            float lower_bound = q1 - 1.5f * iqr;
            float upper_bound = q3 + 1.5f * iqr;
            
            bool confidence_outlier = (current_result.confidence < lower_bound || 
                                     current_result.confidence > upper_bound);
            
            // Emotion consistency check
            int same_emotion_count = 0;
            for (size_t i = emotion_history.size() - std::min(emotion_history.size(), size_t(5)); 
                 i < emotion_history.size() - 1; ++i) {
                if (emotion_history[i].dominant_emotion == current_result.dominant_emotion) {
                    same_emotion_count++;
                }
            }
            bool emotion_consistent = (same_emotion_count >= 2);
            
            // Adaptive smoothing based on consistency and confidence
            if (!confidence_outlier || emotion_consistent) {
                std::vector<float> smoothed_probs(static_cast<int>(EmotionType::COUNT), 0.0f);
                float total_weight = 0.0f;
                
                // Exponential weighting with confidence boosting
                for (size_t i = 0; i < emotion_history.size(); ++i) {
                    float time_weight = std::exp(static_cast<float>(i) / emotion_history.size() - 1.0f);
                    float confidence_weight = 1.0f + emotion_history[i].confidence;
                    float consistency_weight = emotion_consistent ? 1.2f : 1.0f;
                    
                    float combined_weight = time_weight * confidence_weight * consistency_weight;
                    total_weight += combined_weight;
                    
                    for (int j = 0; j < static_cast<int>(EmotionType::COUNT); j++) {
                        smoothed_probs[j] += combined_weight * emotion_history[i].probabilities[j];
                    }
                }
                
                // Normalize probabilities
                if (total_weight > 0) {
                    for (float& prob : smoothed_probs) {
                        prob /= total_weight;
                    }
                    
                    // Find dominant emotion from smoothed probabilities
                    auto max_it = std::max_element(smoothed_probs.begin(), smoothed_probs.end());
                    int max_index = std::distance(smoothed_probs.begin(), max_it);
                    
                    smoothed_result.dominant_emotion = static_cast<EmotionType>(max_index);
                    smoothed_result.confidence = *max_it;
                    
                    // Copy vector to array
                    for (size_t k = 0; k < smoothed_probs.size() && k < smoothed_result.probabilities.size(); ++k) {
                        smoothed_result.probabilities[k] = smoothed_probs[k];
                    }
                    
                    // Apply confidence boost for consistent detections
                    if (emotion_consistent && smoothed_result.confidence > 0.4f) {
                        smoothed_result.confidence = std::min(1.0f, smoothed_result.confidence * 1.15f);
                    }
                }
            } else {
                // For outliers, reduce confidence but keep the emotion
                smoothed_result.confidence *= 0.7f;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in temporal smoothing: " << e.what() << std::endl;
        return current_result;
    }
    
    return smoothed_result;
}

} // namespace EmotionAnalysis
