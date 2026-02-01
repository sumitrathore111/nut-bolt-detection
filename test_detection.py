"""
Test Detection Script - Check what the model detects on a blank/random image
"""

from ultralytics import YOLO
import numpy as np
import cv2

def test_blank_image():
    print("=" * 60)
    print("🧪 TESTING MODEL ON BLANK/RANDOM IMAGES")
    print("=" * 60)
    
    model = YOLO('model/best.pt')
    
    # Test 1: Pure black image
    print("\n📋 Test 1: Pure BLACK image (should detect NOTHING)")
    print("-" * 40)
    black_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(black_img, conf=0.6, verbose=False)
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"   Detections: {detections}")
    if detections > 0:
        print("   ❌ FALSE POSITIVES on black image!")
        for box in results[0].boxes:
            print(f"      - Class: {model.names[int(box.cls[0])]}, Confidence: {float(box.conf[0]):.2f}")
    else:
        print("   ✅ Correct - No false positives")
    
    # Test 2: Pure white image
    print("\n📋 Test 2: Pure WHITE image (should detect NOTHING)")
    print("-" * 40)
    white_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    results = model(white_img, conf=0.6, verbose=False)
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"   Detections: {detections}")
    if detections > 0:
        print("   ❌ FALSE POSITIVES on white image!")
        for box in results[0].boxes:
            print(f"      - Class: {model.names[int(box.cls[0])]}, Confidence: {float(box.conf[0]):.2f}")
    else:
        print("   ✅ Correct - No false positives")
    
    # Test 3: Random noise
    print("\n📋 Test 3: RANDOM NOISE image (should detect NOTHING)")
    print("-" * 40)
    noise_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(noise_img, conf=0.6, verbose=False)
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"   Detections: {detections}")
    if detections > 0:
        print("   ❌ FALSE POSITIVES on noise image!")
        for box in results[0].boxes:
            print(f"      - Class: {model.names[int(box.cls[0])]}, Confidence: {float(box.conf[0]):.2f}")
    else:
        print("   ✅ Correct - No false positives")
    
    # Test 4: Gray gradient
    print("\n📋 Test 4: GRAY GRADIENT image (should detect NOTHING)")
    print("-" * 40)
    gray_img = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(640):
        gray_img[:, i] = i // 3
    results = model(gray_img, conf=0.6, verbose=False)
    detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"   Detections: {detections}")
    if detections > 0:
        print("   ❌ FALSE POSITIVES on gradient image!")
        for box in results[0].boxes:
            print(f"      - Class: {model.names[int(box.cls[0])]}, Confidence: {float(box.conf[0]):.2f}")
    else:
        print("   ✅ Correct - No false positives")
    
    # Test 5: Different confidence thresholds on noise
    print("\n📋 Test 5: Testing CONFIDENCE THRESHOLDS on noise")
    print("-" * 40)
    for conf in [0.25, 0.45, 0.6, 0.7, 0.8]:
        results = model(noise_img, conf=conf, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        status = "❌ FALSE POSITIVES" if detections > 0 else "✅ OK"
        print(f"   Conf={conf}: {detections} detections {status}")
    
    print("\n" + "=" * 60)
    print("📊 RECOMMENDATION:")
    print("=" * 60)
    print("""
If you see false positives above, the issue is with the MODEL itself.
The model may have been:
  1. Trained on insufficient data
  2. Trained without proper negative samples (backgrounds without objects)
  3. Overfitting to training data

To fix this, you need to RETRAIN the model with:
  - More diverse training images
  - Images WITHOUT nuts/bolts (negative samples)
  - Better data augmentation
  
For now, use a HIGHER confidence threshold (0.7 or 0.8) to reduce 
false positives.
""")

if __name__ == "__main__":
    test_blank_image()
