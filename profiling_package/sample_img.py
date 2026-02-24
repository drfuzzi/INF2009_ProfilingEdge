import numpy as np
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and resizes an image to the target dimensions for model profiling.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    
    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize to [0, 1] or [-1, 1] depending on model requirements
    img_array = img_array / 255.0
    
    # Add batch dimension: (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def save_to_header(img_array, filename="image_data.h"):
    """
    Converts the image array to a C header file for embedded C/C++ projects 
    (e.g., for ESP32 or FPGA firmware).
    """
    # Flatten the array for C-style indexing
    flat_data = img_array.flatten()
    
    with open(filename, "w") as f:
        f.write("#ifndef IMAGE_DATA_H\n#define IMAGE_DATA_H\n\n")
        f.write(f"const float image_data[{len(flat_data)}] = {{\n")
        f.write(", ".join(map(str, flat_data)))
        f.write("\n};\n\n#endif")
    print(f"Header file saved as {filename}")

if __name__ == "__main__":
    # Example usage
    INPUT_IMAGE = "test_image.jpg" # Ensure this exists in your repo
    
    try:
        processed_data = preprocess_image(INPUT_IMAGE)
        print(f"Processed image shape: {processed_data.shape}")
        
        # Save as .npy for Python profiling (e.g., on Raspberry Pi/Beelink)
        np.save("sample_input.npy", processed_data)
        
        # Save as .h for Edge device deployment (e.g., ESP32-S3)
        save_to_header(processed_data)
        
    except FileNotFoundError:
        print(f"Error: {INPUT_IMAGE} not found. Please provide a valid image file.")
