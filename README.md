# YOLOv5: Real-Time Object Detection



## 🚀 Overview

Ultralytics YOLOv5 is a cutting-edge, state-of-the-art computer vision model based on the PyTorch framework. YOLOv5 is renowned for its ease of use, speed, and accuracy, making it an excellent choice for a wide range of vision AI tasks including object detection, image segmentation, and image classification.

This repository provides a web interface to easily access and utilize YOLOv5's powerful capabilities without needing to use the command line.

## 🆕 YOLO11: The Next Evolution

We're excited to announce **Ultralytics YOLO11**, the latest advancement in our state-of-the-art vision models! YOLO11 builds on our legacy of speed, precision, and ease of use, delivering exceptional performance for object detection, instance segmentation, pose estimation, image classification, and oriented object detection.

## 🎯 Features

- **Web Interface**: Easy-to-use web interface for all YOLOv5 functionalities
- **Real-time Inference**: Fast object detection and segmentation
- **Multiple Tasks**: Support for detection, segmentation, and classification
- **Pre-trained Models**: Various model sizes from nano to extra-large
- **Custom Training**: Train on your own datasets through the web interface
- **Export Capabilities**: Convert models to various deployment formats

## 🛠️ Web Interface Usage

### Object Detection
Use the web interface to perform object detection on:
- Images
- Videos
- Webcam streams
- Screen captures
- YouTube videos
- RTSP/RTMP streams

### Image Segmentation
Access instance segmentation capabilities through the web interface to:
- Detect and segment objects in images
- Generate pixel-level masks
- Process multiple image formats

### Image Classification
Use the classification features to:
- Classify images into categories
- Support for various datasets (ImageNet, CIFAR, etc.)
- Compare different classification models

## 📊 Model Performance

YOLOv5 offers multiple model sizes to balance speed and accuracy:

| Model | Size | mAP | Speed (V100) | Use Case |
|-------|------|-----|--------------|----------|
| YOLOv5n | 640 | 28.0 | 6.3ms | Mobile/Edge |
| YOLOv5s | 640 | 37.4 | 6.4ms | Balanced |
| YOLOv5m | 640 | 45.4 | 8.2ms | General |
| YOLOv5l | 640 | 49.0 | 10.1ms | High Accuracy |
| YOLOv5x | 640 | 50.7 | 12.1ms | Maximum Accuracy |

## 🚀 Quick Start

### Access the Web Interface
1. Start the web server
2. Open your browser to the provided URL
3. Upload images or videos for processing
4. Configure model settings through the intuitive UI
5. View and download results

### Basic Usage
Through the web interface you can:
- Select from pre-trained models
- Adjust confidence thresholds
- Choose input sources
- View real-time results
- Export processed files

## 🎮 Tasks Supported

### Object Detection
Locate and classify objects in images and videos with bounding boxes.

### Instance Segmentation
Detect objects and generate precise pixel-level masks.

### Image Classification
Categorize entire images into predefined classes.

## 📁 Model Types

### Detection Models
- YOLOv5n.pt, YOLOv5s.pt, YOLOv5m.pt, YOLOv5l.pt, YOLOv5x.pt
- Various sizes for different speed/accuracy requirements

### Segmentation Models  
- YOLOv5n-seg.pt, YOLOv5s-seg.pt, YOLOv5m-seg.pt, YOLOv5l-seg.pt, YOLOv5x-seg.pt
- Combined detection and segmentation

### Classification Models
- YOLOv5n-cls.pt, YOLOv5s-cls.pt, YOLOv5m-cls.pt, YOLOv5l-cls.pt, YOLOv5x-cls.pt
- Specialized for image classification tasks

## 🔧 Advanced Features

### Custom Training
Through the web interface, you can:
- Upload custom datasets
- Configure training parameters
- Monitor training progress
- Evaluate model performance

### Model Export
Export trained models to various formats for deployment:
- ONNX
- TensorRT
- CoreML
- TensorFlow formats

## 📋 Requirements

- Python 3.8 or later
- PyTorch 1.8 or later
- Modern web browser
- Recommended: NVIDIA GPU for faster processing

## 💡 Use Cases

- **Security & Surveillance**: Real-time object detection
- **Autonomous Vehicles**: Environment perception
- **Medical Imaging**: Analysis and detection
- **Retail Analytics**: Customer behavior and inventory
- **Industrial Automation**: Quality control and inspection
- **Research & Development**: Computer vision experiments




<div align="center">

**YOLOv5 Web Interface** - Making computer vision accessible to everyone

</div>
