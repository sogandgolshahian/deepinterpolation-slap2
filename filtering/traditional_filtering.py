import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
import pywt
import os
from pathlib import Path

def apply_gaussian_filter(movie, sigma=1.0):
    """
    Apply Gaussian filter to a movie (3D array).
    
    Args:
        movie (np.ndarray): Input movie as 3D array (frames, height, width)
        sigma (float): Standard deviation for Gaussian kernel
    
    Returns:
        np.ndarray: Filtered movie
    """
    # Convert to float32 for processing
    movie_float = movie.astype(np.float32)
    filtered_movie = np.zeros_like(movie_float)
    for frame in range(movie.shape[0]):
        filtered_movie[frame] = gaussian_filter(movie_float[frame], sigma=sigma)
    return filtered_movie

def apply_wavelet_filter(movie, wavelet='db1', level=1):
    """
    Apply Wavelet transform filter to a movie (3D array).
    
    Args:
        movie (np.ndarray): Input movie as 3D array (frames, height, width)
        wavelet (str): Wavelet type (e.g., 'db1', 'haar', 'sym2')
        level (int): Decomposition level
    
    Returns:
        np.ndarray: Filtered movie
    """
    # Convert to float32 for processing
    movie_float = movie.astype(np.float32)
    filtered_movie = np.zeros_like(movie_float)
    for frame in range(movie.shape[0]):
        # Apply wavelet transform
        coeffs = pywt.wavedec2(movie_float[frame], wavelet, level=level)
        
        # Zero out the detail coefficients
        coeffs_list = list(coeffs)
        for i in range(1, len(coeffs_list)):
            coeffs_list[i] = tuple(np.zeros_like(c) for c in coeffs_list[i])
        
        # Reconstruct the image
        filtered_movie[frame] = pywt.waverec2(coeffs_list, wavelet)
    
    return filtered_movie

# Could create other datatype conversion functions as needed
def convert_to_int16(data, original_data):
    """
    Convert filtered data back to int16 while preserving the original data range.
    
    Args:
        data (np.ndarray): Filtered data in float32
        original_data (np.ndarray): Original data in int16
    
    Returns:
        np.ndarray: Data converted to int16
    """
    # Get the original data range
    data_min = np.min(original_data)
    data_max = np.max(original_data)
    
    # Normalize the filtered data to the original range
    normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    scaled = normalized * (data_max - data_min) + data_min
    
    # Convert to int16
    return np.clip(scaled, data_min, data_max).astype(np.int16)

def process_movie(input_path, output_dir, sigma=1.0, wavelet='db1', level=1):
    """
    Process a movie with both Gaussian and Wavelet filters.
    
    Args:
        input_path (str): Path to input TIF file
        output_dir (str): Directory to save results
        sigma (float): Standard deviation for Gaussian kernel
        wavelet (str): Wavelet type
        level (int): Wavelet decomposition level
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the movie
    print(f"Reading movie: {input_path}")
    movie = tifffile.imread(input_path)
    
    # Verify data type
    if movie.dtype != np.int16:
        print(f"Warning: Input movie is {movie.dtype}, expected int16")
    
    # Apply Gaussian filter
    print("Applying Gaussian filter...")
    gaussian_filtered = apply_gaussian_filter(movie, sigma=sigma)
    gaussian_filtered_int16 = convert_to_int16(gaussian_filtered, movie)
    
    # Apply Wavelet filter
    print("Applying Wavelet filter...")
    wavelet_filtered = apply_wavelet_filter(movie, wavelet=wavelet, level=level)
    wavelet_filtered_int16 = convert_to_int16(wavelet_filtered, movie)
    
    # Save results
    base_name = Path(input_path).stem
    tifffile.imwrite(
        os.path.join(output_dir, f"{base_name}_gaussian_filtered.tif"),
        gaussian_filtered_int16
    )
    tifffile.imwrite(
        os.path.join(output_dir, f"{base_name}_wavelet_filtered.tif"),
        wavelet_filtered_int16
    )
    
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    # Define input and output paths
    input_path = r'C:\path\to\input_file.tif' # Change this
    output_dir = r'C:\path\to\output_folder' # Change this
    
    process_movie(
        input_path=input_path,
        output_dir=output_dir,
        sigma=1.0,  # Adjust this value based on your needs
        wavelet='db1',  # You can try different wavelets: 'haar', 'sym2', etc.
        level=1  # Adjust decomposition level as needed
    ) 
        