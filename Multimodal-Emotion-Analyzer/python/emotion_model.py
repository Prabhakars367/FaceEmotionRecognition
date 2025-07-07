#!/usr/bin/env python3
"""
Advanced Emotion Recognition Model using Deep Learning
Supports TensorFlow/Keras, PyTorch, and ONNX models
Optimized for real-time C++ integration via pybind11
"""

import numpy as np
import cv2
import os
import sys
from typing import List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow available")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch available")
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available")

class EmotionClassifier:
    """
    Advanced emotion classifier supporting multiple deep learning frameworks
    """
    
    EMOTION_LABELS = [
        'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
    ]
    
    def __init__(self, model_path: str, framework: str = 'auto'):
        """
        Initialize emotion classifier
        
        Args:
            model_path: Path to the trained model file
            framework: Framework to use ('tensorflow', 'pytorch', 'onnx', 'auto')
        """
        self.model_path = model_path
        self.framework = framework
        self.model = None
        self.input_size = (48, 48)
        self.is_loaded = False
        
        # Auto-detect framework if not specified
        if framework == 'auto':
            self.framework = self._detect_framework(model_path)
        
        self._load_model()
    
    def _detect_framework(self, model_path: str) -> str:
        """Auto-detect the framework based on file extension"""
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext in ['.h5', '.keras']:
            return 'tensorflow'
        elif ext in ['.pt', '.pth']:
            return 'pytorch'
        elif ext == '.onnx':
            return 'onnx'
        else:
            logger.warning(f"Unknown model format: {ext}, defaulting to TensorFlow")
            return 'tensorflow'
    
    def _load_model(self):
        """Load the emotion recognition model"""
        try:
            if self.framework == 'tensorflow' and TENSORFLOW_AVAILABLE:
                self._load_tensorflow_model()
            elif self.framework == 'pytorch' and PYTORCH_AVAILABLE:
                self._load_pytorch_model()
            elif self.framework == 'onnx' and ONNX_AVAILABLE:
                self._load_onnx_model()
            else:
                raise ValueError(f"Framework {self.framework} not available or not supported")
                
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.framework} model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_loaded = False
    
    def _load_tensorflow_model(self):
        """Load TensorFlow/Keras model"""
        self.model = keras.models.load_model(self.model_path)
        
        # Get input shape
        input_shape = self.model.input_shape
        if len(input_shape) >= 3:
            self.input_size = (input_shape[1], input_shape[2])
        
        logger.info(f"TensorFlow model input size: {self.input_size}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
        
        # Set up transforms for PyTorch
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.Grayscale() if self.input_size == (48, 48) else transforms.ToTensor(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        logger.info(f"PyTorch model loaded with input size: {self.input_size}")
    
    def _load_onnx_model(self):
        """Load ONNX model"""
        self.model = ort.InferenceSession(self.model_path)
        
        # Get input details
        input_details = self.model.get_inputs()[0]
        input_shape = input_details.shape
        
        if len(input_shape) >= 3:
            self.input_size = (input_shape[2], input_shape[3])
        
        logger.info(f"ONNX model loaded with input size: {self.input_size}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for emotion recognition
        
        Args:
            image: Input face image (BGR or grayscale)
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to model input size
        resized = cv2.resize(gray, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Framework-specific preprocessing
        if self.framework == 'tensorflow':
            # Add batch and channel dimensions for TensorFlow
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=-1)  # Add channel dim
            normalized = np.expand_dims(normalized, axis=0)  # Add batch dim
            
        elif self.framework == 'pytorch':
            # PyTorch expects CHW format
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=0)  # Add channel dim
            normalized = np.expand_dims(normalized, axis=0)  # Add batch dim
            
        elif self.framework == 'onnx':
            # ONNX typically expects NCHW format
            if len(normalized.shape) == 2:
                normalized = np.expand_dims(normalized, axis=0)  # Add channel dim
            normalized = np.expand_dims(normalized, axis=0)  # Add batch dim
        
        return normalized
    
    def predict_emotion(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion from face image
        
        Args:
            image: Input face image
            
        Returns:
            Tuple of (predicted_emotion, confidence, probabilities)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Perform inference based on framework
        if self.framework == 'tensorflow':
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
        elif self.framework == 'pytorch':
            with torch.no_grad():
                tensor_input = torch.from_numpy(processed_image)
                predictions = self.model(tensor_input)
                probabilities = torch.softmax(predictions, dim=1).numpy()[0]
                
        elif self.framework == 'onnx':
            input_name = self.model.get_inputs()[0].name
            predictions = self.model.run(None, {input_name: processed_image})
            probabilities = predictions[0][0]
        
        # Get predicted emotion
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = self.EMOTION_LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return predicted_emotion, confidence, probabilities
    
    def predict_emotions_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float, np.ndarray]]:
        """
        Predict emotions for multiple images
        
        Args:
            images: List of face images
            
        Returns:
            List of (predicted_emotion, confidence, probabilities) tuples
        """
        results = []
        for image in images:
            result = self.predict_emotion(image)
            results.append(result)
        return results

class EnhancedEmotionClassifier(EmotionClassifier):
    """
    Enhanced emotion classifier with additional features
    """
    
    def __init__(self, model_path: str, framework: str = 'auto', 
                 confidence_threshold: float = 0.3):
        super().__init__(model_path, framework)
        self.confidence_threshold = confidence_threshold
        self.emotion_history = []
        self.max_history = 10
    
    def predict_emotion_with_smoothing(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Predict emotion with temporal smoothing
        """
        emotion, confidence, probabilities = self.predict_emotion(image)
        
        # Add to history
        self.emotion_history.append((emotion, confidence, probabilities))
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        # Apply temporal smoothing if we have enough history
        if len(self.emotion_history) >= 3:
            # Average probabilities over recent history
            recent_probs = [hist[2] for hist in self.emotion_history[-3:]]
            smoothed_probs = np.mean(recent_probs, axis=0)
            
            # Get smoothed prediction
            smoothed_idx = np.argmax(smoothed_probs)
            smoothed_emotion = self.EMOTION_LABELS[smoothed_idx]
            smoothed_confidence = float(smoothed_probs[smoothed_idx])
            
            return smoothed_emotion, smoothed_confidence, smoothed_probs
        
        return emotion, confidence, probabilities
    
    def get_emotion_distribution(self) -> dict:
        """Get distribution of emotions from recent history"""
        if not self.emotion_history:
            return {}
        
        emotions = [hist[0] for hist in self.emotion_history]
        distribution = {}
        for emotion in self.EMOTION_LABELS:
            distribution[emotion] = emotions.count(emotion) / len(emotions)
        
        return distribution

# Factory function for C++ integration
def create_emotion_classifier(model_path: str, framework: str = 'auto', 
                            enhanced: bool = True) -> Union[EmotionClassifier, EnhancedEmotionClassifier]:
    """
    Factory function to create emotion classifier instance
    
    Args:
        model_path: Path to the model file
        framework: Framework to use ('tensorflow', 'pytorch', 'onnx', 'auto')
        enhanced: Whether to use enhanced classifier with smoothing
        
    Returns:
        Emotion classifier instance
    """
    if enhanced:
        return EnhancedEmotionClassifier(model_path, framework)
    else:
        return EmotionClassifier(model_path, framework)

# Example usage and testing
if __name__ == "__main__":
    # Test the emotion classifier
    import argparse
    
    parser = argparse.ArgumentParser(description="Test emotion recognition model")
    parser.add_argument("--model", required=True, help="Path to emotion model")
    parser.add_argument("--framework", default="auto", choices=["auto", "tensorflow", "pytorch", "onnx"])
    parser.add_argument("--image", help="Test image path")
    parser.add_argument("--camera", action="store_true", help="Use camera for testing")
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = create_emotion_classifier(args.model, args.framework)
    
    if args.image:
        # Test with single image
        image = cv2.imread(args.image)
        if image is not None:
            emotion, confidence, probs = classifier.predict_emotion(image)
            print(f"Predicted emotion: {emotion} (confidence: {confidence:.3f})")
            print(f"All probabilities: {dict(zip(classifier.EMOTION_LABELS, probs))}")
        else:
            print(f"Could not load image: {args.image}")
    
    elif args.camera:
        # Test with camera
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Press 'q' to quit camera test")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extract face
                face = gray[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, _ = classifier.predict_emotion(face)
                
                # Draw results
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion}: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Please specify --image or --camera for testing")
