# Ultralytics 🚀 YOLOv5 Web Interface

import os
import sys
from pathlib import Path
import uuid
from datetime import datetime
import shutil
import base64
import threading
import time

# Fix for Windows MIME types registry issue
import mimetypes
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Fix matplotlib backend compatibility issue
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Workaround for Windows registry access issue
try:
    # Try to initialize MIME types without registry access
    mimetypes.init()
except (KeyboardInterrupt, Exception):
    # If registry access fails, use a minimal MIME types setup
    mimetypes.add_type('image/jpeg', '.jpg')
    mimetypes.add_type('image/jpeg', '.jpeg')
    mimetypes.add_type('image/png', '.png')
    mimetypes.add_type('image/gif', '.gif')
    mimetypes.add_type('image/bmp', '.bmp')
    mimetypes.add_type('video/mp4', '.mp4')
    mimetypes.add_type('video/avi', '.avi')
    mimetypes.add_type('video/quicktime', '.mov')
    mimetypes.add_type('video/x-matroska', '.mkv')
    mimetypes.add_type('video/webm', '.webm')

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, Response
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np

# Setup paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
    check_requirements,
)
from utils.torch_utils import select_device
from ultralytics.utils.plotting import Annotator, colors

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = ROOT / 'uploads'
app.config['RESULTS_FOLDER'] = ROOT / 'results'
app.config['SECRET_KEY'] = 'yolov5-detection-app'

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Global model variable
model = None
device = None
names = None
stride = None

# Camera variables
camera_active = False
camera_thread = None
latest_frame = None
frame_lock = threading.Lock()

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=640):
    """Preprocess image for YOLOv5 inference"""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Convert to tensor
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (scale, (left, top))

def initialize_model(weights='yolov5s.pt', device_id=''):
    """Initialize YOLOv5 model"""
    global model, device, names, stride
    
    try:
        device = select_device(device_id)
        weights_path = ROOT / weights
        
        # Check if weights exist, if not use default
        if not weights_path.exists():
            print(f"Weights {weights} not found, trying default weights...")
            weights_path = ROOT / 'yolov5s.pt'
        
        model = DetectMultiBackend(weights_path, device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
        stride = model.stride
        names = model.names
        
        # Warmup
        model.warmup(imgsz=(1, 3, 640, 640))
        print(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error initializing model: {e}")
        return False

def generate_frames():
    """Generate camera frames with real-time detection"""
    global camera_active, latest_frame, model, device, names
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera started successfully")
    
    while camera_active:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform real-time detection
        if model is not None:
            try:
                # Preprocess frame
                im, (scale, (left, top)) = preprocess_image(frame, 640)
                im = im.to(device)
                
                # Inference
                with torch.no_grad():
                    pred = model(im, augment=False, visualize=False)
                    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)
                
                # Process detections
                annotator = Annotator(frame, line_width=2)
                for det in pred:
                    if len(det):
                        # Scale boxes back to original frame coordinates
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            label = f"{names[c]} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                
                frame = annotator.result()
                
            except Exception as e:
                print(f"Detection error in camera: {e}")
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Update latest frame
        with frame_lock:
            latest_frame = frame_bytes
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.03)
    
    cap.release()
    print("Camera stopped")

def generate_mjpeg():
    """Generate MJPEG stream for live feed"""
    global camera_active, latest_frame
    
    while camera_active:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame
            else:
                # Send a blank frame if no frame available
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# ===========================
# Camera Routes
# ===========================

@app.route('/camera_feed')
def camera_feed():
    """Video streaming route for live camera"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/start', methods=['POST'])
def start_camera():
    """Start the camera for live detection"""
    global camera_active, camera_thread
    
    if camera_active:
        return jsonify({'success': True, 'message': 'Camera already active'})
    
    try:
        camera_active = True
        camera_thread = threading.Thread(target=generate_frames)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Wait a moment for camera to initialize
        time.sleep(1)
        
        return jsonify({
            'success': True, 
            'message': 'Camera started successfully'
        })
    except Exception as e:
        camera_active = False
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the camera"""
    global camera_active, camera_thread
    
    try:
        camera_active = False
        if camera_thread:
            camera_thread.join(timeout=2)
        
        with frame_lock:
            latest_frame = None
        
        return jsonify({
            'success': True, 
            'message': 'Camera stopped successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/camera/status')
def camera_status():
    """Check camera status"""
    return jsonify({
        'active': camera_active,
        'model_loaded': model is not None
    })

@app.route('/camera/snapshot', methods=['POST'])
def take_snapshot():
    """Take a snapshot from the camera with detection results"""
    global latest_frame
    
    if not camera_active or latest_frame is None:
        return jsonify({'success': False, 'error': 'Camera not active or no frame available'}), 400
    
    try:
        with frame_lock:
            if latest_frame is None:
                return jsonify({'success': False, 'error': 'No frame available'}), 400
            
            # Convert frame back to image
            frame_data = latest_frame
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400
            
            # Save snapshot
            snapshot_filename = f"snapshot_{uuid.uuid4().hex[:8]}.jpg"
            snapshot_path = app.config['RESULTS_FOLDER'] / snapshot_filename
            cv2.imwrite(str(snapshot_path), img)
            
            return jsonify({
                'success': True,
                'snapshot_file': snapshot_filename,
                'message': 'Snapshot saved successfully'
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ===========================
# File Detection Routes
# ===========================

def detect_objects(source_path, conf_thres=0.25, iou_thres=0.45, img_size=640):
    """
    Perform object detection on an image or video
    Returns: (result_path, detections_list, file_type, detection_summary, total_objects)
    """
    global model, device, names, stride
    
    if model is None:
        raise Exception("Model not initialized")
    
    # Check if source is image or video
    source_path = Path(source_path)
    file_ext = source_path.suffix.lower()
    is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Create unique result filename
    result_filename = f"{source_path.stem}_{uuid.uuid4().hex[:8]}{'.mp4' if is_video else source_path.suffix}"
    result_path = app.config['RESULTS_FOLDER'] / result_filename
    
    detections = []  # For first frame detections (display in table)
    detection_summary = {}  # Track all detections across all frames
    total_objects = 0  # Total count across all frames
    
    try:
        # Check image size
        imgsz = check_img_size(img_size, s=stride)
        
        if is_video:
            # Process video using OpenCV directly for better control
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {source_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Try different video codecs for better browser compatibility
            fourcc_options = [
                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 - best browser compatibility
                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
                cv2.VideoWriter_fourcc(*'H264'),  # Alternative H.264
                cv2.VideoWriter_fourcc(*'X264'),  # Another H.264 variant
            ]
            
            out = None
            for fourcc_code in fourcc_options:
                try:
                    out = cv2.VideoWriter(str(result_path), fourcc_code, fps, (width, height))
                    if out.isOpened():
                        print(f"✓ Using codec: {fourcc_code}")
                        break
                    else:
                        out = None
                except Exception as e:
                    print(f"✗ Codec {fourcc_code} failed: {e}")
                    continue
            
            # Fallback to any available codec
            if out is None:
                print("⚠ No preferred codec available, using default")
                out = cv2.VideoWriter(str(result_path), 0x7634706d, fps, (width, height))  # mp4v fallback
            
            if not out.isOpened():
                raise ValueError("Could not initialize video writer")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame using the same function as images
                im, (scale, (left, top)) = preprocess_image(frame, imgsz)
                im = im.to(device)
                
                # Inference
                with torch.no_grad():
                    pred = model(im, augment=False, visualize=False)
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
                
                # Process detections
                annotator = Annotator(frame, line_width=2)
                frame_detections = []
                for det in pred:
                    if len(det):
                        # Scale boxes back to original frame coordinates
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            class_name = names[c]
                            label = f"{class_name} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            
                            # Store detection info for this frame
                            detection_info = {
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x) for x in xyxy],
                                'frame': frame_count
                            }
                            frame_detections.append(detection_info)
                            
                            # Update detection summary across all frames
                            detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
                
                # Store detections from first frame for detailed display
                if frame_count == 0:
                    detections.extend(frame_detections)
                
                result_frame = annotator.result()
                
                # Ensure frame is in correct color format for video writing
                if len(result_frame.shape) == 3:
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                out.write(result_frame)
                frame_count += 1
                
                # Progress update every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            out.release()
            
            # Calculate total objects across all frames
            total_objects = sum(detection_summary.values())
            
            print(f"✓ Video processing complete: {frame_count} frames processed")
            print(f"✓ Total objects detected across all frames: {total_objects}")
            print(f"✓ Detection summary: {detection_summary}")
            
        else:
            # Process image
            image = cv2.imread(str(source_path))
            if image is None:
                raise ValueError(f"Could not load image: {source_path}")
            
            # Preprocess image
            im, (scale, (left, top)) = preprocess_image(image, imgsz)
            im = im.to(device)
            
            # Inference
            with torch.no_grad():
                pred = model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
                
                # Process predictions
                annotator = Annotator(image, line_width=3)
                for det in pred:
                    if len(det):
                        # Scale boxes back to original image coordinates
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()
                        
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)
                            class_name = names[c]
                            label = f"{class_name} {conf:.2f}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            
                            detection_info = {
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x) for x in xyxy]
                            }
                            detections.append(detection_info)
                            
                            # Update detection summary for image
                            detection_summary[class_name] = detection_summary.get(class_name, 0) + 1
                
                # Save result
                result_img = annotator.result()
                cv2.imwrite(str(result_path), result_img)
            
            total_objects = len(detections)
            print(f"✓ Image processing complete")
            print(f"✓ Total objects detected: {total_objects}")
            print(f"✓ Detection summary: {detection_summary}")
        
        return str(result_path.name), detections, 'video' if is_video else 'image', detection_summary, total_objects
    
    except Exception as e:
        print(f"Detection error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise e

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
        filepath = app.config['UPLOAD_FOLDER'] / unique_filename
        file.save(str(filepath))
        
        # Get parameters
        conf_thres = float(request.form.get('confidence', 0.25))
        iou_thres = float(request.form.get('iou', 0.45))
        img_size = int(request.form.get('img_size', 640))
        
        # Perform detection
        result = detect_objects(
            filepath,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            img_size=img_size
        )
        
        # Unpack the result
        result_filename, detections, file_type, detection_summary, total_objects = result
        
        return jsonify({
            'success': True,
            'result_file': result_filename,
            'file_type': file_type,
            'detections': detections,
            'summary': detection_summary,
            'total_objects': total_objects
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """Serve result files with proper MIME types"""
    file_path = app.config['RESULTS_FOLDER'] / filename
    
    # Set proper MIME type for videos
    if filename.lower().endswith('.mp4'):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')
    elif filename.lower().endswith(('.avi', '.mov', '.mkv', '.webm')):
        return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')  # Fallback to mp4
    
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/clear', methods=['POST'])
def clear_files():
    """Clear uploaded and result files"""
    try:
        # Clear uploads
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = app.config['UPLOAD_FOLDER'] / file
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Clear results
        for file in os.listdir(app.config['RESULTS_FOLDER']):
            file_path = app.config['RESULTS_FOLDER'] / file
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        return jsonify({'success': True, 'message': 'Files cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized',
        'camera_active': camera_active
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv5 Web Interface')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model weights path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--port', type=int, default=5000, help='port number')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='host address')
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv5 Web Interface")
    print("=" * 60)
    
    # Initialize model
    print(f"\nInitializing model: {args.weights}")
    if initialize_model(args.weights, args.device):
        print(f"\n✓ Server starting on http://{args.host}:{args.port}")
        print(f"✓ Device: {device}")
        print(f"✓ Model: {args.weights}")
        print(f"✓ Camera feature: Available")
        print("\n" + "=" * 60)
        app.run(host=args.host, port=args.port, debug=False)
    else:
        print("✗ Failed to initialize model. Please check your weights file.")