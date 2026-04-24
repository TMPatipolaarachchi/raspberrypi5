#!/bin/bash
# Raspberry Pi 5 Setup Script for Elephant Detection System
# Run this script to install all dependencies and configure the system

set -e

echo "=========================================="
echo "Elephant Detection System Setup"
echo "for Raspberry Pi 5"
echo "=========================================="

# Check if running on Raspberry Pi
if [ ! -f /proc/cpuinfo ] || ! grep -q "Raspberry Pi\|BCM" /proc/cpuinfo; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system
echo ""
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libffi-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev

# Create virtual environment
echo ""
echo "Step 3: Creating Python virtual environment..."
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install Python packages
echo ""
echo "Step 4: Installing Python dependencies..."

# Install NumPy first (specific version for compatibility)
pip install "numpy>=1.21.0,<2.0.0"

# Install TensorFlow for ARM64 (Raspberry Pi 5)
echo "Installing TensorFlow..."
pip install tensorflow-aarch64 2>/dev/null || pip install tensorflow

# Install other dependencies
pip install -r requirements.txt

# Install additional Pi-specific packages
pip install psutil RPi.GPIO

# Create output directories
echo ""
echo "Step 5: Creating output directories..."
mkdir -p output
mkdir -p logs

# Set permissions
chmod +x main.py

# Test import
echo ""
echo "Step 6: Testing installation..."
python3 -c "
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import librosa
import joblib
print('All core dependencies imported successfully!')
print(f'OpenCV version: {cv2.__version__}')
print(f'TensorFlow version: {tf.__version__}')
print(f'NumPy version: {np.__version__}')
"

# Check models
echo ""
echo "Step 7: Checking model files..."
python3 main.py --check-models

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To use the system:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run with video: python main.py <video_path>"
echo "  3. Run with camera: python main.py --camera --preview"
echo ""
echo "For more options: python main.py --help"
echo ""
