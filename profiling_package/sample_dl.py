import torch
import torchvision.models as models
import cv2
import time
import numpy as np

def load_optimized_model(quantize=True):
    """
    Represents Lab 2 & 3: Loading a model optimized for Edge devices.
    """
    if quantize:
        # Quantized models use INT8 weights to save memory and speed up CPU cycles
        print("Using Quantized MobileNetV2 (Int8) - Lab Optimization Active")
        model = models.quantization.mobilenet_v2(weights='DEFAULT', quantize=True)
    else:
        print("Using Standard MobileNetV2 (FP32) - Baseline Model")
        model = models.mobilenet_v2(weights='DEFAULT')
    
    model.eval()
    return model

def run_dl_profiling():
    # Toggle this to compare optimized vs unoptimized in your profiling report
    USE_QUANTIZATION = True 
    
    model = load_optimized_model(quantize=USE_QUANTIZATION)
    
    print("Generating synthetic workload for profiling...")
    
    # 100-frame loop to gather stable hardware metrics
    for i in range(100):
        # Create synthetic 640x480 frame (no camera needed)
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        start = time.perf_counter()

        # Pre-processing: Resize to 224x224 (The size MobileNetV2 expects)
        input_img = cv2.resize(frame, (224, 224))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        
        # Inference: The core workload for 'perf' and 'cProfile' to track
        with torch.no_grad():
            output = model(input_tensor)
            
        end = time.perf_counter()
        fps = 1 / (end - start)
        
        if i % 20 == 0:
            print(f"Frame {i} | Latency: {(end-start)*1000:.2f}ms | FPS: {fps:.2f}")

if __name__ == "__main__":
    run_dl_profiling()
