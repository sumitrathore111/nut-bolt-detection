"""
GPU Performance Test for Nut & Bolt Detection Model
"""
from ultralytics import YOLO
import torch
import time
import numpy as np

def test_performance():
    print("=" * 60)
    print("🚀 GPU PERFORMANCE TEST")
    print("=" * 60)

    # Check CUDA
    print(f"\n🖥️  CUDA Status:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    print("\n📦 Loading model...")
    model = YOLO('model/best.pt')
    print(f"   Model loaded on: {model.device}")

    # Create test image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Test CPU performance
    print("\n⏱️  CPU Performance Test:")
    print("-" * 40)
    model.to('cpu')
    # Warmup
    model(img, device='cpu', verbose=False)

    cpu_times = []
    for i in range(10):
        start = time.time()
        model(img, device='cpu', verbose=False)
        cpu_times.append(time.time() - start)

    cpu_avg = sum(cpu_times) / len(cpu_times) * 1000
    cpu_fps = 1000 / cpu_avg
    print(f"   Average: {cpu_avg:.1f}ms per frame")
    print(f"   FPS: {cpu_fps:.1f}")

    # Test GPU performance
    if torch.cuda.is_available():
        print("\n⚡ GPU Performance Test:")
        print("-" * 40)
        model.to('cuda:0')
        # Warmup (important for GPU!)
        for _ in range(3):
            model(img, device=0, verbose=False)
        torch.cuda.synchronize()

        gpu_times = []
        for i in range(10):
            start = time.time()
            model(img, device=0, verbose=False)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_times.append(time.time() - start)

        gpu_avg = sum(gpu_times) / len(gpu_times) * 1000
        gpu_fps = 1000 / gpu_avg
        print(f"   Average: {gpu_avg:.1f}ms per frame")
        print(f"   FPS: {gpu_fps:.1f}")

        print(f"\n📊 GPU Speedup: {cpu_avg/gpu_avg:.1f}x faster than CPU")

    # Diagnosis
    print("\n" + "=" * 60)
    print("🔍 DIAGNOSIS")
    print("=" * 60)

    if torch.cuda.is_available():
        if gpu_fps < 10:
            print("""
⚠️  LOW FPS DETECTED! Possible causes:

1. 🔄 Model not on GPU during inference
   - The Flask app may be loading model on CPU
   - Need to explicitly move model to GPU

2. 🖼️  Image processing overhead
   - Base64 decoding takes time
   - Image resizing before inference

3. 📡 Network latency (if using webcam stream)
   - Frontend sends frame → Backend processes → Returns result
   - This round-trip adds delay

4. 🔥 GPU Memory
   - RTX 2050 has limited VRAM
   - Batch size of 1 is optimal
""")
        else:
            print(f"✅ GPU performance looks good: {gpu_fps:.1f} FPS")

    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    print("1. Ensure model is loaded to GPU at startup")
    print("2. Increase confidence threshold to 0.6 to reduce processing")
    print("3. Consider reducing image size from 640 to 416 for faster inference")
    print("4. Use Half Precision (FP16) for 2x speedup on RTX GPUs")

if __name__ == "__main__":
    test_performance()
