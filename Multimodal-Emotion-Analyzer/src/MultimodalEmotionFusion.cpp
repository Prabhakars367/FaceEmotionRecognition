#include "MultimodalEmotionFusion.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace EmotionAnalysis {

class MultimodalEmotionFusion::Impl {
public:
    FusionConfig config;
    EmotionResult previous_fused_result;
    bool has_previous_result = false;
};

MultimodalEmotionFusion::MultimodalEmotionFusion() 
    : pImpl(std::make_unique<Impl>()) {}

MultimodalEmotionFusion::~MultimodalEmotionFusion() = default;

void MultimodalEmotionFusion::initialize(const FusionConfig& config) {
    pImpl->config = config;
    
    // Normalize weights
    float total_weight = config.video_weight + config.audio_weight;
    if (total_weight > 0) {
        pImpl->config.video_weight = config.video_weight / total_weight;
        pImpl->config.audio_weight = config.audio_weight / total_weight;
    } else {
        pImpl->config.video_weight = 0.6f;
        pImpl->config.audio_weight = 0.4f;
    }
    
    std::cout << "Multimodal fusion initialized with strategy: " 
              << static_cast<int>(config.strategy) << std::endl;
}

EmotionResult MultimodalEmotionFusion::fuseEmotions(const EmotionResult& video_result, 
                                                    const EmotionResult& audio_result) {
    EmotionResult fused_result;
    
    // Check if we have valid inputs
    bool video_valid = (video_result.confidence >= pImpl->config.confidence_threshold);
    bool audio_valid = (audio_result.confidence >= pImpl->config.confidence_threshold);
    
    if (!video_valid && !audio_valid) {
        // No valid inputs, return unknown emotion
        fused_result.dominant_emotion = EmotionType::UNKNOWN;
        fused_result.confidence = 0.0f;
        return fused_result;
    }
    
    if (!video_valid) {
        // Only audio is valid
        fused_result = audio_result;
    } else if (!audio_valid) {
        // Only video is valid
        fused_result = video_result;
    } else {
        // Both are valid, apply fusion strategy
        switch (pImpl->config.strategy) {
            case FusionStrategy::WEIGHTED_AVERAGE:
                fused_result = weightedAverageFusion(video_result, audio_result);
                break;
                
            case FusionStrategy::MAXIMUM_CONFIDENCE:
                fused_result = maximumConfidenceFusion(video_result, audio_result);
                break;
                
            case FusionStrategy::DYNAMIC_WEIGHTING:
                fused_result = dynamicWeightingFusion(video_result, audio_result);
                break;
                
            case FusionStrategy::TEMPORAL_SMOOTHING:
                // First apply another strategy, then smooth
                fused_result = weightedAverageFusion(video_result, audio_result);
                break;
        }
    }
    
    // Apply temporal smoothing if enabled
    if (pImpl->config.enable_temporal_smoothing && pImpl->has_previous_result) {
        fused_result = temporalSmoothingFusion(fused_result);
    }
    
    // Update previous result
    pImpl->previous_fused_result = fused_result;
    pImpl->has_previous_result = true;
    
    return fused_result;
}

MultimodalResult MultimodalEmotionFusion::createMultimodalResult(const EmotionResult& video_result,
                                                                 const EmotionResult& audio_result) {
    MultimodalResult result;
    
    result.video_emotion = video_result;
    result.audio_emotion = audio_result;
    result.fused_emotion = fuseEmotions(video_result, audio_result);
    result.is_valid = (result.fused_emotion.dominant_emotion != EmotionType::UNKNOWN);
    
    return result;
}

void MultimodalEmotionFusion::updateWeights(float video_weight, float audio_weight) {
    float total = video_weight + audio_weight;
    if (total > 0) {
        pImpl->config.video_weight = video_weight / total;
        pImpl->config.audio_weight = audio_weight / total;
    }
}

void MultimodalEmotionFusion::setFusionStrategy(FusionStrategy strategy) {
    pImpl->config.strategy = strategy;
}

FusionConfig MultimodalEmotionFusion::getConfig() const {
    return pImpl->config;
}

EmotionResult MultimodalEmotionFusion::weightedAverageFusion(const EmotionResult& video_result,
                                                            const EmotionResult& audio_result) {
    EmotionResult result;
    
    // Weighted average of probabilities
    for (size_t i = 0; i < result.probabilities.size(); ++i) {
        result.probabilities[i] = 
            pImpl->config.video_weight * video_result.probabilities[i] +
            pImpl->config.audio_weight * audio_result.probabilities[i];
    }
    
    // Find dominant emotion
    auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
    result.dominant_emotion = static_cast<EmotionType>(
        std::distance(result.probabilities.begin(), max_it)
    );
    
    // Weighted average of confidence
    result.confidence = 
        pImpl->config.video_weight * video_result.confidence +
        pImpl->config.audio_weight * audio_result.confidence;
    
    result.timestamp = std::chrono::steady_clock::now();
    
    return result;
}

EmotionResult MultimodalEmotionFusion::maximumConfidenceFusion(const EmotionResult& video_result,
                                                              const EmotionResult& audio_result) {
    // Return the result with higher confidence
    return (video_result.confidence >= audio_result.confidence) ? video_result : audio_result;
}

EmotionResult MultimodalEmotionFusion::dynamicWeightingFusion(const EmotionResult& video_result,
                                                             const EmotionResult& audio_result) {
    // Enhanced dynamic weighting with context awareness and conflict resolution
    float video_quality = calculateSignalQuality(video_result);
    float audio_quality = calculateSignalQuality(audio_result);
    
    // Check for emotion agreement/disagreement
    bool emotions_agree = (video_result.dominant_emotion == audio_result.dominant_emotion);
    float agreement_bonus = emotions_agree ? 0.2f : 0.0f;
    
    // Emotion-specific modality reliability
    float video_reliability = getModalityReliability(video_result.dominant_emotion, true);
    float audio_reliability = getModalityReliability(audio_result.dominant_emotion, false);
    
    // Calculate enhanced quality scores
    float enhanced_video_quality = video_quality * video_reliability + agreement_bonus;
    float enhanced_audio_quality = audio_quality * audio_reliability + agreement_bonus;
    
    float total_quality = enhanced_video_quality + enhanced_audio_quality;
    if (total_quality == 0) {
        // Fallback to equal weighting
        enhanced_video_quality = enhanced_audio_quality = 0.5f;
        total_quality = 1.0f;
    }
    
    float dynamic_video_weight = enhanced_video_quality / total_quality;
    float dynamic_audio_weight = enhanced_audio_quality / total_quality;
    
    EmotionResult result;
    
    // Apply dynamic weights with conflict resolution
    if (!emotions_agree && std::abs(video_result.confidence - audio_result.confidence) > 0.3f) {
        // Significant disagreement - use conflict resolution
        result = resolveEmotionConflict(video_result, audio_result, dynamic_video_weight, dynamic_audio_weight);
    } else {
        // Normal weighted fusion
        for (size_t i = 0; i < result.probabilities.size(); ++i) {
            result.probabilities[i] = 
                dynamic_video_weight * video_result.probabilities[i] +
                dynamic_audio_weight * audio_result.probabilities[i];
        }
        
        // Find dominant emotion
        auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
        result.dominant_emotion = static_cast<EmotionType>(
            std::distance(result.probabilities.begin(), max_it)
        );
        
        result.confidence = 
            dynamic_video_weight * video_result.confidence +
            dynamic_audio_weight * audio_result.confidence;
    }
    
    result.timestamp = std::chrono::steady_clock::now();
    
    return result;
}

EmotionResult MultimodalEmotionFusion::temporalSmoothingFusion(const EmotionResult& current_result) {
    if (!pImpl->has_previous_result) {
        return current_result;
    }
    
    EmotionResult smoothed_result = current_result;
    float base_alpha = 1.0f - pImpl->config.smoothing_factor;
    
    // Adaptive smoothing based on emotion change magnitude
    float emotion_change = calculateEmotionChange(current_result, pImpl->previous_fused_result);
    float adaptive_alpha = base_alpha;
    
    // Reduce smoothing for significant emotion changes (faster response)
    if (emotion_change > 0.6f) {
        adaptive_alpha = std::min(0.9f, base_alpha + 0.3f);
    } else if (emotion_change < 0.2f) {
        // Increase smoothing for stable emotions (reduce noise)
        adaptive_alpha = std::max(0.3f, base_alpha - 0.2f);
    }
    
    float smoothing_factor = 1.0f - adaptive_alpha;
    
    // Outlier detection - don't smooth if current result is likely an outlier
    bool is_outlier = detectEmotionOutlier(current_result, pImpl->previous_fused_result);
    if (is_outlier && current_result.confidence < 0.7f) {
        // Use more of the previous result if current seems like an outlier
        smoothing_factor = 0.8f;
        adaptive_alpha = 0.2f;
    }
    
    // Smooth probabilities
    for (size_t i = 0; i < smoothed_result.probabilities.size(); ++i) {
        smoothed_result.probabilities[i] = 
            adaptive_alpha * current_result.probabilities[i] +
            smoothing_factor * pImpl->previous_fused_result.probabilities[i];
    }
    
    // Smooth confidence
    smoothed_result.confidence = 
        adaptive_alpha * current_result.confidence +
        smoothing_factor * pImpl->previous_fused_result.confidence;
    
    // Recalculate dominant emotion based on smoothed probabilities
    auto max_it = std::max_element(smoothed_result.probabilities.begin(), 
                                  smoothed_result.probabilities.end());
    smoothed_result.dominant_emotion = static_cast<EmotionType>(
        std::distance(smoothed_result.probabilities.begin(), max_it)
    );
    
    return smoothed_result;
}

float MultimodalEmotionFusion::calculateSignalQuality(const EmotionResult& result) {
    // Calculate signal quality based on confidence and probability distribution
    float quality = result.confidence;
    
    // Penalize if probabilities are too uniform (uncertain)
    float entropy = 0.0f;
    for (float prob : result.probabilities) {
        if (prob > 0) {
            entropy -= prob * std::log2(prob);
        }
    }
    
    // Lower entropy means more certain distribution
    float max_entropy = std::log2(7.0f); // Maximum entropy for 7 emotions
    float certainty = 1.0f - (entropy / max_entropy);
    
    // Combine confidence and certainty
    quality = 0.7f * quality + 0.3f * certainty;
    
    return std::max(0.0f, std::min(1.0f, quality));
}

void MultimodalEmotionFusion::normalizeProbabilities(std::array<float, 7>& probabilities) {
    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    
    if (sum > 0) {
        for (float& prob : probabilities) {
            prob /= sum;
        }
    }
}

// Enhanced conflict resolution for disagreeing modalities
EmotionResult MultimodalEmotionFusion::resolveEmotionConflict(const EmotionResult& video_result,
                                                             const EmotionResult& audio_result,
                                                             float video_weight,
                                                             float audio_weight) {
    EmotionResult result;
    
    // Check for complementary emotions that can coexist
    bool are_complementary = areEmotionsComplementary(video_result.dominant_emotion, 
                                                     audio_result.dominant_emotion);
    
    if (are_complementary) {
        // Blend complementary emotions with increased confidence
        for (size_t i = 0; i < result.probabilities.size(); ++i) {
            result.probabilities[i] = 
                video_weight * video_result.probabilities[i] +
                audio_weight * audio_result.probabilities[i];
        }
        
        // Boost confidence for complementary emotion detection
        auto max_it = std::max_element(result.probabilities.begin(), result.probabilities.end());
        result.confidence = std::min(1.0f, *max_it * 1.2f);
        result.dominant_emotion = static_cast<EmotionType>(
            std::distance(result.probabilities.begin(), max_it)
        );
    } else {
        // For conflicting emotions, use the higher confidence modality but reduce overall confidence
        if (video_result.confidence > audio_result.confidence) {
            result = video_result;
            result.confidence *= 0.8f; // Reduce confidence due to conflict
        } else {
            result = audio_result;
            result.confidence *= 0.8f;
        }
    }
    
    return result;
}

// Check if emotions are complementary (can reasonably coexist)
bool MultimodalEmotionFusion::areEmotionsComplementary(EmotionType emotion1, EmotionType emotion2) {
    // Define complementary emotion pairs
    static const std::vector<std::pair<EmotionType, EmotionType>> complementary_pairs = {
        {EmotionType::SURPRISE, EmotionType::HAPPINESS},
        {EmotionType::SURPRISE, EmotionType::FEAR},
        {EmotionType::ANGER, EmotionType::DISGUST},
        {EmotionType::SADNESS, EmotionType::FEAR},
        {EmotionType::NEUTRAL, EmotionType::HAPPINESS}, // Subtle happiness
        {EmotionType::NEUTRAL, EmotionType::SADNESS}    // Subtle sadness
    };
    
    for (const auto& pair : complementary_pairs) {
        if ((pair.first == emotion1 && pair.second == emotion2) ||
            (pair.first == emotion2 && pair.second == emotion1)) {
            return true;
        }
    }
    
    return false;
}

// Get modality-specific reliability for different emotions
float MultimodalEmotionFusion::getModalityReliability(EmotionType emotion, bool is_video) {
    if (is_video) {
        // Video is generally better for visual emotions
        switch (emotion) {
            case EmotionType::HAPPINESS: return 0.9f;  // Smiles are very visual
            case EmotionType::SURPRISE: return 0.85f;  // Facial expressions clear
            case EmotionType::DISGUST: return 0.8f;    // Facial muscle tension
            case EmotionType::ANGER: return 0.75f;     // Good but can be subtle
            case EmotionType::SADNESS: return 0.7f;    // Sometimes subtle
            case EmotionType::FEAR: return 0.65f;      // Can be confused with surprise
            case EmotionType::NEUTRAL: return 0.6f;    // Baseline reliability
            default: return 0.5f;
        }
    } else {
        // Audio is better for certain emotional expressions
        switch (emotion) {
            case EmotionType::ANGER: return 0.85f;     // Voice tension very clear
            case EmotionType::FEAR: return 0.8f;       // Voice tremor/pitch changes
            case EmotionType::SADNESS: return 0.75f;   // Voice characteristics
            case EmotionType::SURPRISE: return 0.7f;   // Vocal exclamations
            case EmotionType::HAPPINESS: return 0.65f; // Laughter, tone changes
            case EmotionType::DISGUST: return 0.6f;    // Vocal expressions
            case EmotionType::NEUTRAL: return 0.7f;    // Stable baseline
            default: return 0.5f;
        }
    }
}



bool MultimodalEmotionFusion::areEmotionsCompatible(EmotionType emotion1, EmotionType emotion2) {
    // Define emotion compatibility matrix
    if (emotion1 == emotion2) return true;
    
    // Compatible emotion pairs
    std::vector<std::pair<EmotionType, EmotionType>> compatible_pairs = {
        {EmotionType::HAPPINESS, EmotionType::SURPRISE},
        {EmotionType::FEAR, EmotionType::SURPRISE},
        {EmotionType::ANGER, EmotionType::DISGUST},
        {EmotionType::SADNESS, EmotionType::NEUTRAL},
        {EmotionType::NEUTRAL, EmotionType::HAPPINESS}
    };
    
    for (const auto& pair : compatible_pairs) {
        if ((pair.first == emotion1 && pair.second == emotion2) ||
            (pair.first == emotion2 && pair.second == emotion1)) {
            return true;
        }
    }
    
    return false;
}

float MultimodalEmotionFusion::calculateEmotionChange(const EmotionResult& current, 
                                                     const EmotionResult& previous) {
    // Calculate magnitude of change between emotion distributions
    float change = 0.0f;
    
    for (size_t i = 0; i < current.probabilities.size(); ++i) {
        change += std::abs(current.probabilities[i] - previous.probabilities[i]);
    }
    
    // Also consider confidence change
    float confidence_change = std::abs(current.confidence - previous.confidence);
    
    return (change + confidence_change) / 2.0f;
}

bool MultimodalEmotionFusion::detectEmotionOutlier(const EmotionResult& current,
                                                   const EmotionResult& previous) {
    // Detect if current emotion is likely an outlier
    
    // Check for sudden dramatic changes
    float emotion_change = calculateEmotionChange(current, previous);
    if (emotion_change > 0.8f && current.confidence < 0.6f) {
        return true; // Sudden change with low confidence
    }
    
    // Check if dominant emotion changed dramatically with low confidence
    if (current.dominant_emotion != previous.dominant_emotion && 
        current.confidence < 0.5f && previous.confidence > 0.7f) {
        return true;
    }
    
    return false;
}
} // namespace EmotionAnalysis
