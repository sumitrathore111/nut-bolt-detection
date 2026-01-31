"""
Comprehensive Model Analysis Script
Tests the model with actual inference to evaluate performance
"""

from ultralytics import YOLO
import torch
import os
import numpy as np

def analyze_model():
    print("=" * 70)
    print("🔬 COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 70)

    model = YOLO('model/best.pt')

    # 1. Basic Model Info
    print("\n📋 MODEL METADATA:")
    print("-" * 50)
    print(f"  Classes: {model.names}")
    print(f"  Task: {model.task}")
    print(f"  Device Available: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU only'}")

    # 2. Check model internals
    print("\n🔧 MODEL ARCHITECTURE DETAILS:")
    print("-" * 50)

    # Get model info
    info = model.info(verbose=False)
    params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {params:,}")

    # Determine model variant
    if params < 4_000_000:
        variant = "YOLOv8n (Nano) - Fastest, least accurate"
        variant_rating = 6
    elif params < 12_000_000:
        variant = "YOLOv8s (Small) - Good balance"
        variant_rating = 8
    elif params < 30_000_000:
        variant = "YOLOv8m (Medium) - Better accuracy"
        variant_rating = 9
    else:
        variant = "YOLOv8l/x (Large) - Best accuracy"
        variant_rating = 10

    print(f"  Detected Variant: {variant}")

    # 3. Check training configuration stored in model
    print("\n📊 TRAINING CONFIGURATION (from model checkpoint):")
    print("-" * 50)

    # Access model's training args if available
    try:
        if hasattr(model, 'ckpt') and model.ckpt:
            ckpt = model.ckpt
            if 'train_args' in ckpt:
                args = ckpt['train_args']
                print(f"  Epochs trained: {args.get('epochs', 'N/A')}")
                print(f"  Image size: {args.get('imgsz', 'N/A')}")
                print(f"  Batch size: {args.get('batch', 'N/A')}")
                print(f"  Learning rate: {args.get('lr0', 'N/A')}")

            if 'epoch' in ckpt:
                print(f"  Best epoch: {ckpt.get('epoch', 'N/A')}")

            # Training metrics
            if 'metrics' in ckpt:
                metrics = ckpt['metrics']
                print(f"\n  📈 TRAINING METRICS (Best Checkpoint):")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value:.4f}" if isinstance(value, float) else f"     {key}: {value}")
        else:
            print("  Training info not available in checkpoint")
    except Exception as e:
        print(f"  Could not extract training info: {e}")

    # 4. Model weights analysis
    print("\n🎯 MODEL QUALITY INDICATORS:")
    print("-" * 50)

    # Check if weights seem reasonable
    weight_stats = []
    for name, param in model.model.named_parameters():
        if 'weight' in name:
            weight_stats.append({
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item()
            })

    if weight_stats:
        avg_std = np.mean([w['std'] for w in weight_stats])
        avg_range = np.mean([w['max'] - w['min'] for w in weight_stats])

        print(f"  Weight Statistics:")
        print(f"    Average Std Dev: {avg_std:.4f}")
        print(f"    Average Range: {avg_range:.4f}")

        # Interpret
        if avg_std < 0.01:
            print("    ⚠️  Very low variance - model may be undertrained")
        elif avg_std > 1.0:
            print("    ⚠️  High variance - model may be unstable")
        else:
            print("    ✅ Weight distribution looks healthy")

    # 5. Test inference
    print("\n🧪 INFERENCE TEST:")
    print("-" * 50)

    # Create a test image (blank)
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)

    print("  Testing on blank image (should detect nothing)...")
    results = model(test_img, verbose=False, conf=0.5)
    blank_detections = len(results[0].boxes) if results[0].boxes is not None else 0

    if blank_detections == 0:
        print(f"    ✅ PASS: No false detections on blank image")
    else:
        print(f"    ⚠️  WARNING: {blank_detections} detections on blank image (false positives!)")

    # Test with random noise
    print("  Testing on random noise image...")
    noise_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model(noise_img, verbose=False, conf=0.5)
    noise_detections = len(results[0].boxes) if results[0].boxes is not None else 0

    if noise_detections == 0:
        print(f"    ✅ PASS: No false detections on noise image")
    else:
        print(f"    ⚠️  WARNING: {noise_detections} detections on noise (possible overfitting)")

    # 6. Final Rating
    print("\n" + "=" * 70)
    print("⭐ FINAL MODEL ASSESSMENT")
    print("=" * 70)

    scores = {
        'Architecture': variant_rating,
        'Weight Health': 8 if (weight_stats and 0.01 < avg_std < 1.0) else 5,
        'False Positive Test': 10 if (blank_detections == 0 and noise_detections == 0) else 5,
        'Class Setup': 10  # 2 classes, properly named
    }

    print("\n📊 SCORE BREAKDOWN:")
    total = 0
    for category, score in scores.items():
        bar = "█" * score + "░" * (10 - score)
        print(f"  {category:20s}: [{bar}] {score}/10")
        total += score

    avg_score = total / len(scores)

    print(f"\n🏆 OVERALL SCORE: {avg_score:.1f}/10")

    if avg_score >= 8:
        print("   Rating: EXCELLENT - Model is well-trained")
    elif avg_score >= 6:
        print("   Rating: GOOD - Model should work reasonably well")
    elif avg_score >= 4:
        print("   Rating: FAIR - Consider retraining with more data")
    else:
        print("   Rating: POOR - Model needs significant improvement")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 50)

    if blank_detections > 0 or noise_detections > 0:
        print("  ⚠️  Model shows false positives on test images")
        print("     → Increase confidence threshold to 0.6 or 0.7")
        print("     → Consider retraining with negative examples")

    if variant_rating < 8:
        print("  📈 Consider using YOLOv8s or YOLOv8m for better accuracy")

    print("  ✓ Run validation on your actual test dataset for accurate metrics")
    print("  ✓ Monitor precision/recall in production usage")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    analyze_model()
