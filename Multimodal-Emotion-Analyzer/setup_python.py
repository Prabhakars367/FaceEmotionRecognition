#!/usr/bin/env python3
"""
Setup script for Python dependencies in Multimodal Emotion Analyzer
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package):
    """Check if a package is installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    print("Setting up Python environment for Multimodal Emotion Analyzer...")
    
    # Essential packages
    essential_packages = [
        "numpy",
        "opencv-python",
        "pybind11"
    ]
    
    # Optional deep learning packages
    ml_packages = {
        "tensorflow": "TensorFlow",
        "torch": "PyTorch", 
        "onnxruntime": "ONNX Runtime"
    }
    
    print("\nChecking essential packages...")
    for package in essential_packages:
        if check_package(package):
            print(f"✓ {package} is installed")
        else:
            print(f"✗ {package} is missing - installing...")
            if install_package(package):
                print(f"✓ {package} installed successfully")
            else:
                print(f"✗ Failed to install {package}")
    
    print("\nChecking machine learning frameworks...")
    available_frameworks = []
    for package, name in ml_packages.items():
        if check_package(package):
            print(f"✓ {name} is available")
            available_frameworks.append(name)
        else:
            print(f"✗ {name} is not installed")
    
    if not available_frameworks:
        print("\nWarning: No machine learning frameworks detected!")
        print("Install at least one: pip install tensorflow torch onnxruntime")
    else:
        print(f"\nAvailable frameworks: {', '.join(available_frameworks)}")
    
    print("\nSetup complete! You can now build the C++ application with Python integration.")

if __name__ == "__main__":
    main()
