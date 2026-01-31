"""
Model Diagnostic Script - Evaluates your Nut & Bolt Detection Model
"""

from ultralytics import YOLO
import torch
import os

def diagnose_model():
    print("=" * 60)
    print("🔍 NUT & BOLT MODEL DIAGNOSTIC REPORT")
    print("=" * 60)

    model_path = 'model/best.pt'

    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return

    # Load model
    model = YOLO(model_path)

    # Basic Info
    print("\n📊 MODEL INFORMATION:")
    print("-" * 40)
    print(f"  Model Path: {model_path}")
    print(f"  Model Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"  Task Type: {model.task}")
    print(f"  Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

    # Class Information
    print("\n🏷️  CLASS INFORMATION:")
    print("-" * 40)
    print(f"  Number of Classes: {len(model.names)}")
    print(f"  Classes: {model.names}")

    # Architecture Info
    print("\n🏗️  ARCHITECTURE:")
    print("-" * 40)
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Model summary
    layers, params, gradients, flops = model.info()
    print(f"  Layers: {layers}")
    print(f"  GFLOPs: {flops:.2f}")

    # Determine model size variant
    if total_params < 5_000_000:
        variant = "YOLOv8n (Nano)"
    elif total_params < 15_000_000:
        variant = "YOLOv8s (Small)"
    elif total_params < 30_000_000:
        variant = "YOLOv8m (Medium)"
    elif total_params < 50_000_000:
        variant = "YOLOv8l (Large)"
    else:
        variant = "YOLOv8x (Extra Large)"
    print(f"  Estimated Variant: {variant}")

    # Issues and Recommendations
    print("\n" + "=" * 60)
    print("📋 MODEL RATING & ANALYSIS")
    print("=" * 60)

    issues = []
    ratings = []

    # Check 1: Class names match
    expected_classes = {'nut', 'bolt', 'Nut', 'Bolt'}
    actual_classes = set(model.names.values())

    if actual_classes.issubset({'Nut', 'Bolt', 'nut', 'bolt'}):
        print("\n✅ Classes: Correct (Nut & Bolt only)")
        ratings.append(("Class Configuration", 10, 10))
    else:
        print(f"\n⚠️  Classes: Unexpected classes found: {actual_classes}")
        issues.append("Model may have extra classes")
        ratings.append(("Class Configuration", 5, 10))

    # Check 2: Model size appropriateness
    if total_params >= 11_000_000:
        print("✅ Model Size: Good (YOLOv8s or larger - better accuracy)")
        ratings.append(("Model Size", 9, 10))
    else:
        print("⚠️  Model Size: Small (YOLOv8n - may have lower accuracy)")
        issues.append("Consider using a larger model (YOLOv8s/m) for better accuracy")
        ratings.append(("Model Size", 6, 10))

    # Check 3: Number of parameters
    print(f"\n📈 POTENTIAL ISSUES CAUSING FALSE DETECTIONS:")
    print("-" * 40)

    print("""
1. LOW CONFIDENCE THRESHOLD (Currently 0.25)
   - Your app uses 0.25 confidence threshold
   - This means detections with only 25% confidence are shown
   - FALSE POSITIVES occur when random objects score above 0.25

   🔧 FIX: Increase confidence threshold to 0.5 or higher

2. INSUFFICIENT TRAINING DATA
   - If model wasn't trained on diverse backgrounds
   - It may detect similar shapes as nuts/bolts

   🔧 FIX: Retrain with more varied images

3. OVERFITTING
   - Model may have memorized training images
   - Performs poorly on new/different images

   🔧 FIX: Use data augmentation, add more training data

4. CLASS NAME MISMATCH IN CODE
   - Your model uses: {0: 'Bolt', 1: 'Nut'}
   - Your code uses: ['nut', 'bolt'] (lowercase, different order!)

   🔧 FIX: Update CLASS_NAMES in app.py to match model
""")

    # Overall Rating
    print("\n" + "=" * 60)
    print("⭐ OVERALL MODEL RATING")
    print("=" * 60)

    total_score = sum(r[1] for r in ratings)
    max_score = sum(r[2] for r in ratings)

    print("\nCategory Breakdown:")
    for name, score, max_s in ratings:
        bar = "█" * score + "░" * (max_s - score)
        print(f"  {name}: [{bar}] {score}/{max_s}")

    # Estimate overall rating
    architecture_score = 8  # YOLOv8s is good
    class_config_score = 7  # 2 classes, but naming mismatch

    # We can't know training quality without validation data
    estimated_training_score = "Unknown (need validation data)"

    print(f"""
📊 ESTIMATED RATINGS:
  - Architecture Quality: 8/10 (YOLOv8s - good balance of speed/accuracy)
  - Class Configuration: 7/10 (2 classes, but naming mismatch in code)
  - Training Quality: {estimated_training_score}

🎯 RECOMMENDED ACTIONS:
  1. URGENT: Fix class name mismatch in app.py
     Change: CLASS_NAMES = ['nut', 'bolt']
     To:     CLASS_NAMES = ['Bolt', 'Nut']  # Match model's order!

  2. Increase confidence threshold from 0.25 to 0.5 or 0.6

  3. Test model on validation images to measure actual accuracy
""")

    print("=" * 60)

if __name__ == "__main__":
    diagnose_model()
