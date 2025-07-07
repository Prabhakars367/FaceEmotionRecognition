#pragma once

#include "EmotionTypes.hpp"
#include "VideoEmotionAnalyzer.hpp"
#include "AudioEmotionAnalyzer.hpp"
#include <memory>
#include <functional>

namespace EmotionAnalysis {

enum class FusionStrategy {
    WEIGHTED_AVERAGE,    // Simple weighted average
    MAXIMUM_CONFIDENCE,  // Choose result with highest confidence
    DYNAMIC_WEIGHTING,   // Adjust weights based on signal quality
    TEMPORAL_SMOOTHING   // Consider temporal consistency
};

struct FusionConfig {
    FusionStrategy strategy = FusionStrategy::WEIGHTED_AVERAGE;
    float video_weight = 0.6f;      // Weight for video emotion
    float audio_weight = 0.4f;      // Weight for audio emotion
    float confidence_threshold = 0.3f; // Minimum confidence to consider result
    bool enable_temporal_smoothing = true;
    float smoothing_factor = 0.7f;   // For temporal smoothing (0.0 = no smoothing, 1.0 = max smoothing)
};

class MultimodalEmotionFusion {
public:
    MultimodalEmotionFusion();
    ~MultimodalEmotionFusion();
    
    // Initialize with fusion configuration
    void initialize(const FusionConfig& config = FusionConfig{});
    
    // Fuse video and audio emotion results
    EmotionResult fuseEmotions(const EmotionResult& video_result, 
                              const EmotionResult& audio_result);
    
    // Get multimodal result combining video, audio, and fusion
    MultimodalResult createMultimodalResult(const EmotionResult& video_result,
                                           const EmotionResult& audio_result);
    
    // Update fusion weights dynamically
    void updateWeights(float video_weight, float audio_weight);
    
    // Set fusion strategy
    void setFusionStrategy(FusionStrategy strategy);
    
    // Get current fusion configuration
    FusionConfig getConfig() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Fusion strategy implementations
    EmotionResult weightedAverageFusion(const EmotionResult& video_result,
                                       const EmotionResult& audio_result);
    
    EmotionResult maximumConfidenceFusion(const EmotionResult& video_result,
                                         const EmotionResult& audio_result);
    
    EmotionResult dynamicWeightingFusion(const EmotionResult& video_result,
                                        const EmotionResult& audio_result);
    
    EmotionResult temporalSmoothingFusion(const EmotionResult& current_result);
    
    // Helper methods
    float calculateSignalQuality(const EmotionResult& result);
    void normalizeProbabilities(std::array<float, 7>& probabilities);
    
    // Enhanced fusion helper methods
    float getModalityReliability(EmotionType emotion, bool is_video);
    EmotionResult resolveEmotionConflict(const EmotionResult& video_result,
                                        const EmotionResult& audio_result,
                                        float video_weight, float audio_weight);
    bool areEmotionsCompatible(EmotionType emotion1, EmotionType emotion2);
    bool areEmotionsComplementary(EmotionType emotion1, EmotionType emotion2);
    float calculateEmotionChange(const EmotionResult& current, const EmotionResult& previous);
    bool detectEmotionOutlier(const EmotionResult& current, const EmotionResult& previous);
};

} // namespace EmotionAnalysis
