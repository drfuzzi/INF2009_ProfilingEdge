import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def process_for_edge_profiling(image_path, downscale_factor=2):
    """
    Simulates the Image Analytics lab workflow:
    1. Loads a real image via OpenCV.
    2. Converts to Grayscale (as required by HoG).
    3. Downscales (Crucial for Edge performance).
    4. Extracts HoG features for a 'feature-based' profiling test.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # 1. Grayscale Conversion (Standard for analytics)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Downscaling (The 'Informed Decision' from Lab 5)
    # Reduces compute load for Edge devices like ESP32/Raspberry Pi
    new_size = (gray.shape[1] // downscale_factor, gray.shape[0] // downscale_factor)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    # 3. HoG Feature Extraction (From Analytics Lab 5)
    # This represents the actual 'workload' we want to profile
    features, hog_image = hog(resized, orientations=8, pixels_per_cell=(16, 16),
                               cells_per_block=(1, 1), visualize=True)
    
    # Rescale HoG image for visualization/saving
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return resized, features, hog_image_rescaled

def export_to_header(data, var_name, filename="edge_data.h"):
    """
    Exports the processed analytics data into a C header for the Profiling Lab.
    """
    flat_data = data.flatten()
    with open(filename, "w") as f:
        f.write(f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n")
        f.write(f"const float {var_name}[{len(flat_data)}] = {{\n")
        
        # Write 8 values per line for readability
        for i in range(0, len(flat_data), 8):
            f.write("    " + ", ".join([f"{x:.4f}f" for x in flat_data[i:i+8]]) + ",\n")
            
        f.write("};\n\n#endif")
    print(f"✅ Exported {var_name} to {filename}")

if __name__ == "__main__":
    INPUT_FILE = "test_image.jpg" # Student's image from previous lab
    
    try:
        # Process the image using analytics techniques
        img_data, feature_data, viz = process_for_edge_profiling(INPUT_FILE)
        
        print(f"Original Image -> Processed Shape: {img_data.shape}")
        print(f"Extracted {len(feature_data)} HoG features for profiling.")

        # Export for Edge Profiling
        # We export the processed image AND the features to see which is 
        # faster to process on the hardware
        export_to_header(img_data, "input_image_gray", "image_input.h")
        export_to_header(feature_data, "hog_features", "features_input.h")

    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Ensure 'test_image.jpg' from the Analytics lab is in this folder.")
