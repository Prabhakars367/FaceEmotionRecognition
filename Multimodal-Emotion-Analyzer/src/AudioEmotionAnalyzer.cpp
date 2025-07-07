#include "AudioEmotionAnalyzer.hpp"
#ifndef NO_AUDIO
#include <portaudio.h>
#endif
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>

namespace EmotionAnalysis {

#ifndef NO_AUDIO

class AudioEmotionAnalyzer::Impl {
public:
    AudioConfig config;
    PaStream* stream = nullptr;
    std::atomic<bool> is_capturing{false};
    std::atomic<float> current_level{0.0f};
    
    std::mutex result_mutex;
    EmotionResult latest_result;
    
    std::function<void(const EmotionResult&)> emotion_callback;
    
    std::mutex buffer_mutex;
    std::queue<std::vector<float>> audio_buffers;
    std::thread processing_thread;
    std::atomic<bool> should_stop_processing{false};
    
    // Audio callback
    static int audioCallback(const void* input, void* output,
                           unsigned long frameCount,
                           const PaStreamCallbackTimeInfo* timeInfo,
                           PaStreamCallbackFlags statusFlags,
                           void* userData);
    
    void processAudioBuffers();
    
    // Feature extraction and classification methods for Impl class
    std::vector<float> extractFeatures(const std::vector<float>& audio_data);
    EmotionResult classifyAudioEmotion(const std::vector<float>& features);
    float calculateZeroCrossingRate(const std::vector<float>& audio);
    float calculateSpectralCentroid(const std::vector<float>& audio);
    std::vector<float> calculateMFCCLikeFeatures(const std::vector<float>& audio);
    float calculateSpectralRolloff(const std::vector<float>& audio, float percentile = 0.85f);
    float calculateSpectralFlux(const std::vector<float>& audio);
    std::vector<float> calculateFormantFrequencies(const std::vector<float>& audio);
};

// Static audio callback function
int AudioEmotionAnalyzer::Impl::audioCallback(const void* input, void* output,
                                              unsigned long frameCount,
                                              const PaStreamCallbackTimeInfo* timeInfo,
                                              PaStreamCallbackFlags statusFlags,
                                              void* userData) {
    
    auto* analyzer_impl = static_cast<Impl*>(userData);
    const float* input_buffer = static_cast<const float*>(input);
    
    if (input_buffer && analyzer_impl->is_capturing) {
        // Copy audio data
        std::vector<float> audio_data(input_buffer, input_buffer + frameCount);
        
        // Calculate current audio level
        float level = 0.0f;
        for (float sample : audio_data) {
            level += std::abs(sample);
        }
        level /= frameCount;
        analyzer_impl->current_level.store(level);
        
        // Add to processing queue
        {
            std::lock_guard<std::mutex> lock(analyzer_impl->buffer_mutex);
            analyzer_impl->audio_buffers.push(std::move(audio_data));
            
            // Limit queue size to prevent memory issues
            while (analyzer_impl->audio_buffers.size() > 10) {
                analyzer_impl->audio_buffers.pop();
            }
        }
    }
    
    return paContinue;
}

void AudioEmotionAnalyzer::Impl::processAudioBuffers() {
    std::vector<float> accumulated_audio;
    const size_t target_samples = static_cast<size_t>(config.sample_rate * config.duration_seconds);
    
    while (!should_stop_processing) {
        std::vector<float> audio_data;
        
        // Get audio data from queue
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            if (!audio_buffers.empty()) {
                audio_data = std::move(audio_buffers.front());
                audio_buffers.pop();
            }
        }
        
        if (!audio_data.empty()) {
            // Accumulate audio data
            accumulated_audio.insert(accumulated_audio.end(), 
                                   audio_data.begin(), audio_data.end());
            
            // Process when we have enough data
            if (accumulated_audio.size() >= target_samples) {
                // Keep only the most recent samples
                if (accumulated_audio.size() > target_samples) {
                    accumulated_audio.erase(accumulated_audio.begin(), 
                                          accumulated_audio.end() - target_samples);
                }
                
                // Actually analyze the emotion from the audio data
                std::vector<float> features = this->extractFeatures(accumulated_audio);
                EmotionResult result = this->classifyAudioEmotion(features);
                
                // Update latest result
                {
                    std::lock_guard<std::mutex> lock(result_mutex);
                    latest_result = result;
                }
                
                // Call callback if set
                if (emotion_callback) {
                    emotion_callback(result);
                }
                
                // Clear buffer for next analysis
                accumulated_audio.clear();
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

AudioEmotionAnalyzer::AudioEmotionAnalyzer() 
    : pImpl(std::make_unique<Impl>()) {}

AudioEmotionAnalyzer::~AudioEmotionAnalyzer() {
    stopCapture();
    
    if (pImpl->stream) {
        Pa_CloseStream(pImpl->stream);
    }
    Pa_Terminate();
}

bool AudioEmotionAnalyzer::initialize(const AudioConfig& config) {
    pImpl->config = config;
    
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    
    PaStreamParameters input_params;
    input_params.device = Pa_GetDefaultInputDevice();
    if (input_params.device == paNoDevice) {
        std::cerr << "Error: No default input device." << std::endl;
        return false;
    }
    
    input_params.channelCount = config.channels;
    input_params.sampleFormat = paFloat32;
    input_params.suggestedLatency = Pa_GetDeviceInfo(input_params.device)->defaultLowInputLatency;
    input_params.hostApiSpecificStreamInfo = nullptr;
    
    err = Pa_OpenStream(&pImpl->stream,
                        &input_params,
                        nullptr, // no output
                        config.sample_rate,
                        config.frames_per_buffer,
                        paClipOff,
                        &Impl::audioCallback,
                        pImpl.get());
    
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    
    std::cout << "Audio emotion analyzer initialized successfully." << std::endl;
    return true;
}

bool AudioEmotionAnalyzer::startCapture() {
    if (!pImpl->stream) {
        std::cerr << "Audio stream not initialized." << std::endl;
        return false;
    }
    
    PaError err = Pa_StartStream(pImpl->stream);
    if (err != paNoError) {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    
    pImpl->is_capturing = true;
    pImpl->should_stop_processing = false;
    
    // Start processing thread
    pImpl->processing_thread = std::thread(&Impl::processAudioBuffers, pImpl.get());
    
    std::cout << "Audio capture started." << std::endl;
    return true;
}

bool AudioEmotionAnalyzer::stopCapture() {
    if (pImpl->stream && pImpl->is_capturing) {
        pImpl->is_capturing = false;
        pImpl->should_stop_processing = true;
        
        if (pImpl->processing_thread.joinable()) {
            pImpl->processing_thread.join();
        }
        
        PaError err = Pa_StopStream(pImpl->stream);
        if (err != paNoError) {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        
        std::cout << "Audio capture stopped." << std::endl;
    }
    return true;
}

EmotionResult AudioEmotionAnalyzer::getLatestResult() const {
    std::lock_guard<std::mutex> lock(pImpl->result_mutex);
    return pImpl->latest_result;
}

void AudioEmotionAnalyzer::setEmotionCallback(std::function<void(const EmotionResult&)> callback) {
    pImpl->emotion_callback = std::move(callback);
}

float AudioEmotionAnalyzer::getCurrentAudioLevel() const {
    return pImpl->current_level.load();
}

EmotionResult AudioEmotionAnalyzer::analyzeAudioBuffer(const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        return EmotionResult{};
    }
    
    try {
        // Extract features using the Impl class
        std::vector<float> features = pImpl->extractFeatures(audio_data);
        
        // Classify emotion using the Impl class
        EmotionResult result = pImpl->classifyAudioEmotion(features);
        
        // Update latest result
        {
            std::lock_guard<std::mutex> lock(pImpl->result_mutex);
            pImpl->latest_result = result;
        }
        
        // Call callback if set
        if (pImpl->emotion_callback) {
            pImpl->emotion_callback(result);
        }
        
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing audio: " << e.what() << std::endl;
        return EmotionResult{};
    }
}

std::vector<float> AudioEmotionAnalyzer::Impl::extractFeatures(const std::vector<float>& audio_data) {
    std::vector<float> features;
    
    if (audio_data.empty()) {
        return features;
    }
    
    // Enhanced audio features for superior emotion analysis
    
    // 1. Zero Crossing Rate - measures speech clarity and voicing
    float zcr = calculateZeroCrossingRate(audio_data);
    features.push_back(zcr);
    
    // 2. Short-time Energy - measures vocal intensity
    float energy = 0.0f;
    for (float sample : audio_data) {
        energy += sample * sample;
    }
    energy /= audio_data.size();
    features.push_back(energy);
    
    // 3. RMS (Root Mean Square) - vocal power
    float rms = std::sqrt(energy);
    features.push_back(rms);
    
    // 4. Enhanced Spectral Centroid - brightness/sharpness of voice
    float spectral_centroid = calculateSpectralCentroid(audio_data);
    features.push_back(spectral_centroid);
    
    // 5. Spectral Rolloff - frequency distribution characteristic
    float spectral_rolloff = calculateSpectralRolloff(audio_data);
    features.push_back(spectral_rolloff);
    
    // 6. Spectral Flux - measure of spectral change rate
    float spectral_flux = calculateSpectralFlux(audio_data);
    features.push_back(spectral_flux);
    
    // 7. MFCC-like features for timbral characteristics
    std::vector<float> mfcc_features = calculateMFCCLikeFeatures(audio_data);
    features.insert(features.end(), mfcc_features.begin(), mfcc_features.end());
    
    // 8. Formant frequencies for vowel characteristics
    std::vector<float> formants = calculateFormantFrequencies(audio_data);
    features.insert(features.end(), formants.begin(), formants.end());
    
    // 9. Enhanced pitch variation (Jitter) - voice stability
    float pitch_var = 0.0f;
    if (audio_data.size() > 10) {
        for (size_t i = 10; i < audio_data.size(); i += 10) {
            float current_segment = 0.0f;
            float prev_segment = 0.0f;
            
            for (size_t j = i-10; j < i; ++j) {
                prev_segment += std::abs(audio_data[j]);
            }
            for (size_t j = i; j < std::min(i+10, audio_data.size()); ++j) {
                current_segment += std::abs(audio_data[j]);
            }
            
            pitch_var += std::abs(current_segment - prev_segment);
        }
        pitch_var /= (audio_data.size() / 10);
    }
    features.push_back(pitch_var);
    
    // 10. Voice Activity Detection (VAD) - speech vs silence ratio
    float vad_ratio = 0.0f;
    float adaptive_threshold = rms * 0.15f; // More sophisticated threshold
    int voice_samples = 0;
    for (float sample : audio_data) {
        if (std::abs(sample) > adaptive_threshold) {
            voice_samples++;
        }
    }
    vad_ratio = static_cast<float>(voice_samples) / audio_data.size();
    features.push_back(vad_ratio);
    
    // 11. Temporal dynamics with multiple time scales
    float short_term_dynamics = 0.0f;
    float long_term_dynamics = 0.0f;
    
    if (audio_data.size() > 20) {
        // Short-term dynamics (5-sample windows)
        for (size_t i = 5; i < audio_data.size(); i += 5) {
            float current_energy = 0.0f;
            float prev_energy = 0.0f;
            
            for (size_t j = i-5; j < i; ++j) {
                prev_energy += audio_data[j] * audio_data[j];
            }
            for (size_t j = i; j < std::min(i+5, audio_data.size()); ++j) {
                current_energy += audio_data[j] * audio_data[j];
            }
            
            short_term_dynamics += std::abs(current_energy - prev_energy);
        }
        short_term_dynamics /= (audio_data.size() / 5);
        
        // Long-term dynamics (20-sample windows)
        for (size_t i = 20; i < audio_data.size(); i += 20) {
            float current_energy = 0.0f;
            float prev_energy = 0.0f;
            
            for (size_t j = i-20; j < i; ++j) {
                prev_energy += audio_data[j] * audio_data[j];
            }
            for (size_t j = i; j < std::min(i+20, audio_data.size()); ++j) {
                current_energy += audio_data[j] * audio_data[j];
            }
            
            long_term_dynamics += std::abs(current_energy - prev_energy);
        }
        long_term_dynamics /= (audio_data.size() / 20);
    }
    features.push_back(short_term_dynamics);
    features.push_back(long_term_dynamics);
    
    return features;
}

EmotionResult AudioEmotionAnalyzer::Impl::classifyAudioEmotion(const std::vector<float>& features) {
    EmotionResult result;
    
    if (features.size() < 12) { // Updated for new feature count
        return result; // Not enough features
    }
    
    // Enhanced research-based emotion classification using comprehensive audio features
    
    float zcr = features[0];                    // Zero crossing rate
    float energy = features[1];                 // Short-time energy
    float rms = features[2];                    // RMS power
    float spectral_centroid = features[3];      // Spectral centroid
    float spectral_rolloff = features[4];       // Spectral rolloff
    float spectral_flux = features[5];          // Spectral flux
    // MFCC-like features are at indices 6-8
    float mfcc1 = features.size() > 6 ? features[6] : 0.0f;
    float mfcc2 = features.size() > 7 ? features[7] : 0.0f;
    float mfcc3 = features.size() > 8 ? features[8] : 0.0f;
    // Formant frequencies are at indices 9-10
    float formant1 = features.size() > 9 ? features[9] : 0.0f;
    float formant2 = features.size() > 10 ? features[10] : 0.0f;
    // Remaining features
    size_t base_idx = 11;
    float pitch_var = features.size() > base_idx ? features[base_idx] : 0.0f;
    float vad_ratio = features.size() > base_idx + 1 ? features[base_idx + 1] : 0.0f;
    float short_term_dynamics = features.size() > base_idx + 2 ? features[base_idx + 2] : 0.0f;
    float long_term_dynamics = features.size() > base_idx + 3 ? features[base_idx + 3] : 0.0f;
    
    // Normalize features for better classification
    float energy_norm = std::min(energy * 1000.0f, 1.0f);
    float rms_norm = std::min(rms * 100.0f, 1.0f);
    float zcr_norm = std::min(zcr * 10.0f, 1.0f);
    float flux_norm = std::min(spectral_flux * 50.0f, 1.0f);
    
    // Calculate emotion scores using enhanced feature analysis
    float happiness_score = 0.0f;
    float sadness_score = 0.0f;
    float anger_score = 0.0f;
    float fear_score = 0.0f;
    float surprise_score = 0.0f;
    float neutral_score = 0.0f;
    float disgust_score = 0.0f;
    
    // HAPPINESS: High energy, high spectral centroid, moderate jitter, good formant definition
    if (energy_norm > 0.25f && spectral_centroid > 280.0f && vad_ratio > 0.4f) {
        happiness_score += 0.35f;
        if (mfcc2 > 0.3f && formant1 > 200.0f) happiness_score += 0.25f; // Clear vowel articulation
        if (short_term_dynamics > 0.001f) happiness_score += 0.2f; // Dynamic variation
        if (pitch_var > 0.008f && pitch_var < 0.025f) happiness_score += 0.2f; // Controlled variation
    }
    
    // SADNESS: Low energy, low spectral features, minimal dynamics, lower formants
    if (energy_norm < 0.15f && spectral_centroid < 220.0f && vad_ratio < 0.6f) {
        sadness_score += 0.4f;
        if (mfcc1 < 0.2f && formant1 < 180.0f) sadness_score += 0.2f; // Muffled speech
        if (long_term_dynamics < 0.0003f) sadness_score += 0.2f; // Monotonous
        if (pitch_var < 0.006f) sadness_score += 0.2f; // Flat intonation
    }
    
    // ANGER: Very high energy, high spectral features, high flux, strong formants
    if (energy_norm > 0.45f && spectral_centroid > 350.0f && flux_norm > 0.3f) {
        anger_score += 0.4f;
        if (spectral_rolloff > 0.7f) anger_score += 0.2f; // High-frequency emphasis
        if (pitch_var > 0.02f) anger_score += 0.2f; // High variation
        if (vad_ratio > 0.7f && short_term_dynamics > 0.002f) anger_score += 0.2f; // Intense speech
    }
    
    // FEAR: High ZCR, irregular dynamics, medium energy, tremulous characteristics
    if (zcr_norm > 0.4f && short_term_dynamics > 0.0015f && energy_norm > 0.15f) {
        fear_score += 0.35f;
        if (pitch_var > 0.015f && spectral_flux > 0.01f) fear_score += 0.25f; // Tremulous voice
        if (formant2 > formant1 * 1.8f) fear_score += 0.2f; // Constricted vocal tract
        if (vad_ratio > 0.3f && vad_ratio < 0.7f) fear_score += 0.2f; // Hesitant speech
    }
    
    // SURPRISE: Sudden energy changes, high spectral flux, abrupt formant transitions
    if (long_term_dynamics > 0.0008f && flux_norm > 0.25f && energy_norm > 0.2f) {
        surprise_score += 0.4f;
        if (spectral_rolloff > 0.6f && mfcc3 > 0.25f) surprise_score += 0.2f; // Bright exclamations
        if (pitch_var > 0.012f && short_term_dynamics > 0.0018f) surprise_score += 0.25f; // Rapid changes
        if (zcr_norm > 0.35f) surprise_score += 0.15f; // Crisp articulation
    }
    
    // DISGUST: Specific spectral characteristics, constrained articulation
    if (energy_norm > 0.1f && energy_norm < 0.35f && spectral_centroid > 200.0f && spectral_centroid < 350.0f) {
        disgust_score += 0.3f;
        if (mfcc2 < 0.25f && formant1 < 200.0f) disgust_score += 0.25f; // Constrained articulation
        if (pitch_var > 0.006f && pitch_var < 0.015f) disgust_score += 0.2f; // Controlled variation
        if (spectral_flux > 0.005f && spectral_flux < 0.02f) disgust_score += 0.25f; // Moderate flux
    }
    
    // NEUTRAL: Balanced characteristics across all features
    if (energy_norm > 0.08f && energy_norm < 0.4f && 
        pitch_var > 0.004f && pitch_var < 0.018f && 
        vad_ratio > 0.25f && vad_ratio < 0.8f) {
        neutral_score += 0.35f;
        if (spectral_centroid > 180.0f && spectral_centroid < 320.0f) neutral_score += 0.25f; // Normal range
        if (short_term_dynamics < 0.0015f && long_term_dynamics < 0.0006f) neutral_score += 0.2f; // Steady
        if (mfcc1 > 0.15f && mfcc1 < 0.4f) neutral_score += 0.2f; // Normal timber
    }
    
    // Find the dominant emotion
    std::vector<std::pair<EmotionType, float>> emotion_scores = {
        {EmotionType::HAPPINESS, happiness_score},
        {EmotionType::SADNESS, sadness_score},
        {EmotionType::ANGER, anger_score},
        {EmotionType::FEAR, fear_score},
        {EmotionType::SURPRISE, surprise_score},
        {EmotionType::DISGUST, disgust_score},
        {EmotionType::NEUTRAL, neutral_score}
    };
    
    // Sort by score
    std::sort(emotion_scores.begin(), emotion_scores.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Set the result with improved confidence calculation
    result.dominant_emotion = emotion_scores[0].first;
    
    // Calculate confidence based on separation between top emotions
    float top_score = emotion_scores[0].second;
    float second_score = emotion_scores[1].second;
    float score_separation = top_score - second_score;
    result.confidence = std::min(top_score + score_separation * 0.5f, 1.0f);
    
    // Set probabilities (normalize so they sum to 1)
    float total_score = 0.0f;
    for (const auto& pair : emotion_scores) {
        total_score += std::max(0.01f, pair.second); // Ensure minimum probability
    }
    
    if (total_score > 0.0f) {
        for (const auto& pair : emotion_scores) {
            int idx = static_cast<int>(pair.first);
            result.probabilities[idx] = std::max(0.01f, pair.second) / total_score;
        }
    } else {
        // Default to neutral if no clear emotion detected
        result.dominant_emotion = EmotionType::NEUTRAL;
        result.probabilities[static_cast<int>(EmotionType::NEUTRAL)] = 1.0f;
        result.confidence = 0.5f;
    }
    
    // Ensure minimum confidence for reliability
    if (result.confidence < 0.25f) {
        result.dominant_emotion = EmotionType::NEUTRAL;
        result.confidence = 0.5f;
        // Reset probabilities
        for (int i = 0; i < 7; ++i) {
            result.probabilities[i] = 0.0f;
        }
        result.probabilities[static_cast<int>(EmotionType::NEUTRAL)] = 1.0f;
    }
    
    result.timestamp = std::chrono::steady_clock::now();
    return result;
}

float AudioEmotionAnalyzer::Impl::calculateZeroCrossingRate(const std::vector<float>& audio) {
    if (audio.size() < 2) return 0.0f;
    
    int crossings = 0;
    for (size_t i = 1; i < audio.size(); ++i) {
        if ((audio[i-1] >= 0) != (audio[i] >= 0)) {
            crossings++;
        }
    }
    
    return static_cast<float>(crossings) / (audio.size() - 1);
}

float AudioEmotionAnalyzer::Impl::calculateSpectralCentroid(const std::vector<float>& audio) {
    // Simplified spectral centroid calculation
    // In a real implementation, you'd use FFT
    
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;
    
    for (size_t i = 0; i < audio.size(); ++i) {
        float magnitude = std::abs(audio[i]);
        weighted_sum += i * magnitude;
        magnitude_sum += magnitude;
    }
    
    return (magnitude_sum > 0) ? weighted_sum / magnitude_sum : 0.0f;
}

std::vector<float> AudioEmotionAnalyzer::Impl::calculateMFCCLikeFeatures(const std::vector<float>& audio) {
    std::vector<float> mfcc_features;
    
    if (audio.size() < 32) {
        mfcc_features.resize(3, 0.0f);
        return mfcc_features;
    }
    
    // Simplified MFCC-like features using mel-scale approximation
    // Real MFCC would use DCT of log mel-filter bank outputs
    
    // Divide audio into frequency bands (approximating mel-scale)
    size_t band_size = audio.size() / 8;
    std::vector<float> band_energies(8, 0.0f);
    
    for (size_t band = 0; band < 8; ++band) {
        size_t start = band * band_size;
        size_t end = std::min(start + band_size, audio.size());
        
        for (size_t i = start; i < end; ++i) {
            band_energies[band] += audio[i] * audio[i];
        }
        band_energies[band] /= (end - start);
        
        // Apply log transform (mel-scale approximation)
        band_energies[band] = std::log(band_energies[band] + 1e-10f);
    }
    
    // Calculate first 3 MFCC-like coefficients
    mfcc_features.push_back(band_energies[0] + band_energies[1] + band_energies[2]); // Low frequencies
    mfcc_features.push_back(band_energies[3] + band_energies[4] - band_energies[0] - band_energies[1]); // Mid-low contrast
    mfcc_features.push_back(band_energies[5] + band_energies[6] + band_energies[7]); // High frequencies
    
    return mfcc_features;
}

float AudioEmotionAnalyzer::Impl::calculateSpectralRolloff(const std::vector<float>& audio, float percentile) {
    if (audio.empty()) return 0.0f;
    
    // Calculate total spectral energy
    float total_energy = 0.0f;
    for (float sample : audio) {
        total_energy += std::abs(sample);
    }
    
    // Find rolloff frequency
    float rolloff_threshold = total_energy * percentile;
    float cumulative_energy = 0.0f;
    
    for (size_t i = 0; i < audio.size(); ++i) {
        cumulative_energy += std::abs(audio[i]);
        if (cumulative_energy >= rolloff_threshold) {
            return static_cast<float>(i) / audio.size();
        }
    }
    
    return 1.0f; // All frequencies
}

float AudioEmotionAnalyzer::Impl::calculateSpectralFlux(const std::vector<float>& audio) {
    if (audio.size() < 4) return 0.0f;
    
    // Calculate spectral flux as sum of positive differences between consecutive frames
    float flux = 0.0f;
    size_t frame_size = 16; // Small frame for flux calculation
    
    for (size_t i = frame_size; i < audio.size(); i += frame_size) {
        float current_energy = 0.0f;
        float prev_energy = 0.0f;
        
        // Current frame energy
        for (size_t j = i; j < std::min(i + frame_size, audio.size()); ++j) {
            current_energy += std::abs(audio[j]);
        }
        
        // Previous frame energy
        for (size_t j = i - frame_size; j < i; ++j) {
            prev_energy += std::abs(audio[j]);
        }
        
        // Add positive difference (spectral flux)
        float diff = current_energy - prev_energy;
        if (diff > 0) {
            flux += diff;
        }
    }
    
    return flux / (audio.size() / frame_size);
}

std::vector<float> AudioEmotionAnalyzer::Impl::calculateFormantFrequencies(const std::vector<float>& audio) {
    std::vector<float> formants;
    
    if (audio.size() < 64) {
        formants.resize(2, 100.0f); // Default formant values
        return formants;
    }
    
    // Simplified formant estimation using spectral peak detection
    // Real formant analysis would use Linear Prediction Coding (LPC)
    
    // Divide spectrum into regions where formants typically occur
    size_t f1_region_start = static_cast<size_t>(audio.size() * 0.05); // ~250-800 Hz region
    size_t f1_region_end = static_cast<size_t>(audio.size() * 0.15);
    size_t f2_region_start = static_cast<size_t>(audio.size() * 0.15); // ~800-2500 Hz region
    size_t f2_region_end = static_cast<size_t>(audio.size() * 0.4);
    
    // Find peak in F1 region
    float f1_max = 0.0f;
    size_t f1_peak_idx = f1_region_start;
    for (size_t i = f1_region_start; i < f1_region_end && i < audio.size(); ++i) {
        float magnitude = std::abs(audio[i]);
        if (magnitude > f1_max) {
            f1_max = magnitude;
            f1_peak_idx = i;
        }
    }
    
    // Find peak in F2 region
    float f2_max = 0.0f;
    size_t f2_peak_idx = f2_region_start;
    for (size_t i = f2_region_start; i < f2_region_end && i < audio.size(); ++i) {
        float magnitude = std::abs(audio[i]);
        if (magnitude > f2_max) {
            f2_max = magnitude;
            f2_peak_idx = i;
        }
    }
    
    // Convert indices to approximate frequencies (normalized)
    float f1_freq = static_cast<float>(f1_peak_idx) * 22050.0f / audio.size(); // Assuming ~44kHz sample rate
    float f2_freq = static_cast<float>(f2_peak_idx) * 22050.0f / audio.size();
    
    formants.push_back(f1_freq);
    formants.push_back(f2_freq);
    
    return formants;
}
#else // NO_AUDIO defined - fallback implementation

class AudioEmotionAnalyzer::Impl {
public:
    AudioConfig config;
    EmotionResult latest_result;
    std::function<void(const EmotionResult&)> emotion_callback;
    
    bool initialize() { return true; }
    void startCapture() { 
        std::cout << "Audio capture disabled (PortAudio not available)" << std::endl;
    }
    void stopCapture() {}
    void shutdown() {}
    void setEmotionCallback(std::function<void(const EmotionResult&)> callback) {
        emotion_callback = callback;
    }
    EmotionResult getLatestResult() const { return latest_result; }
    bool isCapturing() const { return false; }
    float getCurrentAudioLevel() const { return 0.0f; }
};

AudioEmotionAnalyzer::AudioEmotionAnalyzer() : pImpl(std::make_unique<Impl>()) {}
AudioEmotionAnalyzer::~AudioEmotionAnalyzer() = default;

bool AudioEmotionAnalyzer::initialize(const AudioConfig& config) {
    pImpl->config = config;
    return pImpl->initialize();
}

void AudioEmotionAnalyzer::startCapture() { pImpl->startCapture(); }
void AudioEmotionAnalyzer::stopCapture() { pImpl->stopCapture(); }
void AudioEmotionAnalyzer::shutdown() { pImpl->shutdown(); }

void AudioEmotionAnalyzer::setEmotionCallback(std::function<void(const EmotionResult&)> callback) {
    pImpl->setEmotionCallback(callback);
}

EmotionResult AudioEmotionAnalyzer::getLatestResult() const {
    return pImpl->getLatestResult();
}

bool AudioEmotionAnalyzer::isCapturing() const {
    return pImpl->isCapturing();
}

float AudioEmotionAnalyzer::getCurrentAudioLevel() const {
    return pImpl->getCurrentAudioLevel();
}

#endif // NO_AUDIO

} // namespace EmotionAnalysis
