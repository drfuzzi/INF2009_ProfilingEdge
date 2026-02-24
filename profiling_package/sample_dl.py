import torch
import torchvision.models as models
import cv2
import time
import numpy as np

def load_model(quantized=False):
    """
    Loads MobileNetV2. 
    If quantized=True, it loads the pre-quantized version for Edge performance.
    """
    if quantized:
        print("Loading Quantized MobileNetV2 (Int8)...")
        # Load the quantized version from torchvision
        model = models.quantization.mobilenet_v2(weights='DEFAULT', quantize=True)
    else:
        print("Loading Standard MobileNetV2 (FP32)...")
        model = models.mobilenet_v2(weights='DEFAULT')
    
    model.eval()
    return model

def preprocess_frame(frame):
    """
    Resizes and normalizes the camera frame to 224x224 for MobileNetV2.
    """
    # Resize to model input size
    input_img = cv2.resize(frame, (224, 224))
    # Convert BGR to RGB and normalize to [0, 1]
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
    # Convert to Tensor and add batch dimension (1, 3, 224, 224)
    input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().unsqueeze(0)
    return input_tensor

def run_inference():
    # SETTING: Change this to True to see the 'Quantization FPS Boost'
    quantize = False 
    
    model = load_model(quantized=quantize)
    cap = cv2.VideoCapture(0)
    
    print("Starting Real-time Inference. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()
        
        # 1. Preprocess
        input_tensor = preprocess_frame(frame)
        
        # 2. Inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # 3. Calculate Performance
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # Display results on frame
        cv2.putText(frame, f"FPS: {fps:.2f} (Quantized: {quantize})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('DL on Edge Profiling', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
