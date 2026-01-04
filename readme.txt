"""
OPTIMIZED CUBE SYMBOL DETECTOR FOR RASPBERRY PI 3
==================================================

Performance Target: 10-200ms per frame
Accuracy Target: >85% on real-world data
Hardware: Raspberry Pi 3 (1GB RAM, ARM CPU)

Architecture:
    1. YOLOv5n: Detect cube boxes (50-80ms)
    2. TFLite: Classify 30 symbols (12ms)
    3. Total: 50-120ms ✓

Author: Optimized for production use
"""
Note: 
best(1).pt
best.tflite (suitable for Raspberry Pi3 nhé)