import cv2
import numpy as np

def process_and_save_image(image_path, output_filename="processed_edge.jpg", downscale_factor=2):
    """
    1. Loads or generates a synthetic image.
    2. Converts to Grayscale and downscales for Edge efficiency.
    3. Runs HoG feature extraction.
    4. Saves the resulting pre-processed image to a JPG file.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ {image_path} not found. Generating synthetic image...")
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # 1. Grayscale Conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Downscaling (The 'Informed Decision' to reduce CPU cycles)
    new_size = (gray.shape[1] // downscale_factor, gray.shape[0] // downscale_factor)
    resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    # 3. Save the processed image to JPG
    # This allows students to visually inspect the downscaled/gray version
    cv2.imwrite(output_filename, resized)
    print(f"✅ Processed image saved to: {output_filename}")

    # 4. HoG Feature Extraction (OpenCV Implementation)
    win_size = (resized.shape[1] // 8 * 8, resized.shape[0] // 8 * 8)
    resized_hog = cv2.resize(resized, win_size)
    
    hog = cv2.HOGDescriptor(_winSize=win_size,
                            _blockSize=(16, 16),
                            _blockStride=(8, 8),
                            _cellSize=(8, 8),
                            _nbins=9)
    
    features = hog.compute(resized_hog)
    print(f"Extracted {len(features)} HoG features.")

if __name__ == "__main__":
    process_and_save_image("test_image.jpg")
