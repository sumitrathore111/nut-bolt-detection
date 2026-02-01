"""
Nut & Bolt Detection Backend API
Flask server that loads a YOLOv8 model and provides detection endpoints.
"""

import os
import io
import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'best.pt')
CONFIDENCE_THRESHOLD = 0.5  # Balanced threshold
IOU_THRESHOLD = 0.45
INPUT_SIZE = 640
USE_GPU = True  # Set to False to force CPU
USE_HALF = False  # FP16 disabled - causes dtype issues with this model

# Size filters disabled - using model confidence only
MAX_BOX_RATIO = 1.0  # No limit
MIN_BOX_SIZE = 5     # Very small minimum
MAX_BOX_SIZE = 2000  # Very large maximum (effectively disabled)

# Class names - MUST match model's class order: {0: 'Bolt', 1: 'Nut'}
CLASS_NAMES = ['Bolt', 'Nut']  # Fixed: matches model's class indices
CLASS_COLORS = {
    'Bolt': (0, 191, 255),     # Deep Sky Blue
    'Nut': (255, 165, 0),      # Orange
}


def load_model():
    """Load the YOLOv8 model from the model directory and move to GPU."""
    global model
    try:
        import torch
        from ultralytics import YOLO

        if os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)

            # Move model to GPU if available
            if USE_GPU and torch.cuda.is_available():
                model.to('cuda:0')
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")

                # Warmup the model for consistent performance
                print("🔥 Warming up GPU...")
                import numpy as np
                dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
                for _ in range(3):
                    model(dummy, verbose=False)
                print("✅ GPU warmup complete!")
            else:
                print("⚠️ GPU not available, using CPU (slower)")
                print("Model device:", model.device)

            return True
        else:
            print(f"⚠️ Model file not found at: {MODEL_PATH}")
            print("Please place your 'best.pt' file in the 'model' folder.")
            return False
    except ImportError:
        print("❌ Error: ultralytics package not installed.")
        print("Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def decode_image(image_data):
    """Decode base64 image data to numpy array."""
    try:
        # Handle data URL format (data:image/jpeg;base64,...)
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array (BGR for OpenCV compatibility)
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        return img_array
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def run_detection(image):
    """Run object detection on the image."""
    global model

    if model is None:
        return None, "Model not loaded"

    try:
        import torch

        # Run inference (model is already on GPU from load_model)
        results = model(
            image,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=INPUT_SIZE,
            half=USE_HALF and torch.cuda.is_available(),  # Use FP16 for GPU
            verbose=False
        )

        detections = []

        # Process results
        for result in results:
            boxes = result.boxes

            # Debug: print number of raw detections
            if boxes is not None and len(boxes) > 0:
                print(f"🔍 Found {len(boxes)} detection(s) with conf >= {CONFIDENCE_THRESHOLD}")

            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Calculate box dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    img_height, img_width = image.shape[:2]
                    img_area = img_width * img_height

                    # Filter 1: Skip if box is too large (like a cup or large object)
                    if box_width > MAX_BOX_SIZE or box_height > MAX_BOX_SIZE:
                        print(f"⚠️ Skipping: Box too large ({box_width:.0f}x{box_height:.0f} > {MAX_BOX_SIZE})")
                        continue

                    # Filter 2: Skip if box is too small (noise)
                    if box_width < MIN_BOX_SIZE or box_height < MIN_BOX_SIZE:
                        print(f"⚠️ Skipping: Box too small ({box_width:.0f}x{box_height:.0f} < {MIN_BOX_SIZE})")
                        continue

                    # Filter 3: Skip if box takes up too much of the image
                    box_ratio = box_area / img_area
                    if box_ratio > MAX_BOX_RATIO:
                        print(f"⚠️ Skipping: Box ratio too large ({box_ratio:.2f} > {MAX_BOX_RATIO})")
                        continue

                    # Get confidence and class
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # Get class name
                    if class_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES[class_id]
                    else:
                        # Use model's class names if available
                        class_name = model.names.get(class_id, f'class_{class_id}')

                    # Get color for this class
                    color = CLASS_COLORS.get(class_name, (0, 255, 0))

                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x2': round(x2, 2),
                            'y2': round(y2, 2)
                        },
                        'color': color
                    })

        return detections, None

    except Exception as e:
        return None, str(e)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API and model status."""
    return jsonify({
        'status': 'online',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'class_names': CLASS_NAMES,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'input_size': INPUT_SIZE
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint.
    Accepts: JSON with base64 encoded image
    Returns: JSON with detection results
    """
    start_time = time.time()

    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please check server logs.',
            'detections': []
        }), 503

    # Get JSON data
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({
            'success': False,
            'error': 'No image data provided. Send JSON with "image" field containing base64 data.',
            'detections': []
        }), 400

    # Decode image
    image = decode_image(data['image'])

    if image is None:
        return jsonify({
            'success': False,
            'error': 'Failed to decode image. Ensure valid base64 format.',
            'detections': []
        }), 400

    # Run detection
    detections, error = run_detection(image)

    if error:
        return jsonify({
            'success': False,
            'error': f'Detection failed: {error}',
            'detections': []
        }), 500

    # Calculate processing time
    processing_time = round((time.time() - start_time) * 1000, 2)  # in milliseconds

    # Count by class
    counts = {}
    for det in detections:
        class_name = det['class']
        counts[class_name] = counts.get(class_name, 0) + 1

    return jsonify({
        'success': True,
        'detections': detections,
        'counts': counts,
        'total': len(detections),
        'processing_time_ms': processing_time,
        'image_size': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    })


@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    return jsonify({
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'input_size': INPUT_SIZE,
        'class_names': CLASS_NAMES,
        'class_colors': CLASS_COLORS
    })


@app.route('/config', methods=['POST'])
def update_config():
    """Update detection configuration."""
    global CONFIDENCE_THRESHOLD, IOU_THRESHOLD

    data = request.get_json()

    if 'confidence_threshold' in data:
        CONFIDENCE_THRESHOLD = float(data['confidence_threshold'])

    if 'iou_threshold' in data:
        IOU_THRESHOLD = float(data['iou_threshold'])

    return jsonify({
        'success': True,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD
    })


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': [
            'GET /health - Check API status',
            'POST /detect - Run detection on image',
            'GET /config - Get configuration',
            'POST /config - Update configuration'
        ]
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': str(e)
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔩 NUT & BOLT DETECTION API SERVER")
    print("="*60 + "\n")

    # Load the model on startup
    model_loaded = load_model()

    if not model_loaded:
        print("\n⚠️ WARNING: Model not loaded!")
        print("The server will start but detections will fail.")
        print(f"Please place your model file at: {MODEL_PATH}\n")

    print("\n📡 Starting server...")
    print("🌐 API will be available at: http://localhost:5000")
    print("📋 Endpoints:")
    print("   - GET  /health  - Check API status")
    print("   - POST /detect  - Run detection")
    print("   - GET  /config  - Get configuration")
    print("   - POST /config  - Update configuration")
    print("\n" + "="*60 + "\n")

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
