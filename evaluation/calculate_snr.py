import numpy as np
import tifffile
import os
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
import pandas as pd
import glob

def format_title(name):
    words = name.split('_')
    return ' '.join(word.capitalize() for word in words)

def calculate_snr_all_regions(frame):
    # Shift data to be positive
    frame = frame - frame.min() + 1  # Add 1 to avoid zero values; could remove this based on your data
    
    threshold = filters.threshold_otsu(frame)
    binary_mask = frame > threshold
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=100)
    cleaned_mask = morphology.remove_small_holes(cleaned_mask, area_threshold=100)

    labeled_mask = measure.label(cleaned_mask)
    props = measure.regionprops(labeled_mask, intensity_image=frame)

    if len(props) == 0:
        return 0.0

    signal_pixels = np.concatenate([p.intensity_image[p.image] for p in props])
    signal_mean = np.mean(signal_pixels)

    background_pixels = frame[~cleaned_mask]
    if len(background_pixels) == 0:
        return 0.0
        
    noise_std = np.std(background_pixels)

    if noise_std <= 0 or signal_mean <= 0:
        return 0.0

    snr = signal_mean / noise_std
    return 20 * np.log10(snr)

def process_movie_snr(movie_path):
    print(f"Processing {os.path.basename(movie_path)}...")
    movie = tifffile.imread(movie_path)
    
    # Process all frames
    snr_values = []
    for frame in movie:
        snr = calculate_snr_all_regions(frame)
        snr_values.append(snr)
    
    # Calculate statistics excluding zeros
    valid_snr = [x for x in snr_values if x > 0]
    if len(valid_snr) > 0:
        snr_mean = np.mean(valid_snr)
        snr_std = np.std(valid_snr)
    else:
        snr_mean = 0
        snr_std = 0
    
    return snr_mean, snr_std, snr_values

def plot_snr(snr_values, movie_name, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values)
    
    # Calculate average excluding zeros
    valid_snr = [x for x in snr_values if x > 0]
    if len(valid_snr) > 0:
        avg_snr = np.mean(valid_snr)
        plt.axhline(y=avg_snr, color='r', linestyle='--', alpha=0.7)
        plt.text(0.02, 0.98, 
                f'Average SNR: {avg_snr:.2f} dB\nValid frames: {len(valid_snr)}/{len(snr_values)}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'SNR per Frame - {format_title(movie_name)}')
    plt.xlabel('Frame')
    plt.ylabel('SNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{movie_name}_snr.png'))
    plt.close()

def main():
    folders = [
        r'C:\path\to\folder1',
        r'C:\path\to\folder2',
        # Add/remove folders/files as needed
    ]
    output_dir = r'C:\path\to\output_folder' # Change this
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = []

    for folder in folders:
        tif_files = glob.glob(os.path.join(folder, '*.tif'))
        for tif_file in tif_files:
            movie_name = os.path.basename(tif_file).replace('.tif', '')
            snr_mean, snr_std, snr_values = process_movie_snr(tif_file)
            
            metrics = {
                'movie_name': movie_name,
                'snr_mean': snr_mean,
                'snr_std': snr_std,
                'valid_frames': len([x for x in snr_values if x > 0]),
                'total_frames': len(snr_values)
            }
            all_metrics.append(metrics)
            plot_snr(snr_values, movie_name, output_dir)

    # Save metrics to CSV
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(output_dir, 'snr_summary.csv'), index=False)

    # Create bar plot for averages
    plt.figure(figsize=(12, 6))
    formatted_names = [format_title(name) for name in df['movie_name']]
    plt.bar(formatted_names, df['snr_mean'], yerr=df['snr_std'])
    plt.title('Average SNR per Movie')
    plt.xlabel('Movie')
    plt.ylabel('SNR (dB)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_snr.png'))
    plt.close()

    print("\nSummary:")
    print(df.to_string())
    print(f"\nResults saved in {output_dir}")

if __name__ == "__main__":
    main()
