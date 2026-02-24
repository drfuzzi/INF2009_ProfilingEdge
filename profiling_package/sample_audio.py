import librosa
import numpy as np
import time

def extract_edge_features(audio_path, sample_rate=16000):
    """
    Simulates the Sound Analytics lab:
    1. Loads audio at 16kHz (The 'Edge Standard' from Lab 4).
    2. Extracts MFCCs (The 'Fingerprint' from Lab 5).
    3. Flattens for Edge Profiling.
    """
    print(f"Loading {audio_path} at {sample_rate}Hz...")
    
    # Load audio - Lab 4 benchmarking shows 16k is better for Edge efficiency
    y, sr = librosa.load(audio_path, sr=sample_rate)

    # Extract MFCCs - Lab 5 explains this is the 'texture/timbre' of the sound
    # Using 13 coefficients as it's lightweight for microcontrollers
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Calculate the mean MFCC across time to create a fixed-size feature vector
    # This vector represents the sound in just 13 numbers!
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return mfcc_mean

def export_to_header(feature_vector, var_name, filename="audio_features.h"):
    """
    Exports the sound 'fingerprint' to a C header for the Profiling Lab.
    """
    with open(filename, "w") as f:
        f.write(f"#ifndef {var_name.upper()}_H\n#define {var_name.upper()}_H\n\n")
        f.write(f"// Extracted MFCC Feature Vector (13 coefficients)\n")
        f.write(f"const float {var_name}[13] = {{\n")
        f.write("    " + ", ".join([f"{x:.4f}f" for x in feature_vector]) + "\n")
        f.write("};\n\n#endif")
    print(f"✅ Success: Exported {var_name} to {filename}")

if __name__ == "__main__":
    # Ensure you have the 'test.wav' from the Hardware Testing section of the lab
    INPUT_AUDIO = "test.wav" 
    
    try:
        # 1. Feature Extraction (Representing the Analytics Lab)
        features = extract_edge_features(INPUT_AUDIO)
        print(f"Extracted MFCC Vector: {features}")
        
        # 2. Export for Profiling (Representing the Edge Lab)
        export_to_header(features, "audio_fingerprint")
        
    except FileNotFoundError:
        print(f"Error: {INPUT_AUDIO} not found.")
        print("Tip: Run 'arecord --duration=5 test.wav' first as per Lab Section 3.")
