import numpy as np
import tifffile
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

def calculate_ssim(clean_frame, noisy_frame, denoised_frame=None):
    """Calculate SSIM for a single frame"""
    metrics = {
        'ssim': ssim(clean_frame, noisy_frame)
    }
    
    if denoised_frame is not None:
        metrics.update({
            'ssim_denoised': ssim(clean_frame, denoised_frame)
        })
    
    return metrics

def process_ssim_for_movie(clean_path, noisy_path, denoised_path=None):
    """Process a movie and calculate SSIM metrics"""
    print(f"Loading movies...")
    start_time = time.time()
    
    # Load movies
    clean_movie = tifffile.imread(clean_path)
    noisy_movie = tifffile.imread(noisy_path)
    
    # Handle denoised movie if provided
    if denoised_path:
        denoised_movie = tifffile.imread(denoised_path)
    else:
        denoised_movie = None
    
    load_time = time.time() - start_time
    print(f"Movies loaded in {load_time:.2f} seconds")
    
    # Calculate metrics for each frame
    print("Calculating SSIM for each frame...")
    frame_metrics = []
    total_frames = len(clean_movie)
    
    for frame_idx in range(total_frames):
        if frame_idx % 50 == 0:  # Update every 50 frames
            elapsed = time.time() - start_time
            frames_per_second = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - (frame_idx + 1)
            estimated_remaining_time = remaining_frames / frames_per_second if frames_per_second > 0 else 0
            print(f"Processing frame {frame_idx+1}/{total_frames} ({((frame_idx+1)/total_frames*100):.1f}%) - "
                  f"Est. remaining time: {estimated_remaining_time:.1f} seconds")
        
        metrics = calculate_ssim(
            clean_movie[frame_idx],
            noisy_movie[frame_idx],
            denoised_movie[frame_idx] if denoised_movie is not None else None
        )
        frame_metrics.append(metrics)
    
    # Calculate averages and standard deviations
    print("Calculating summary statistics...")
    summary = {}
    for metric in frame_metrics[0].keys():
        values = [m[metric] for m in frame_metrics]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)
    
    total_time = time.time() - start_time
    print(f"Processing completed in {total_time:.2f} seconds")
    
    return summary, frame_metrics

def plot_ssim_over_time(frame_metrics, movie_name, output_dir):
    """Create and save plots showing SSIM values over time"""
    print(f"Creating SSIM plot for {movie_name}...")
    
    plt.figure(figsize=(12, 6))
    
    # SSIM plot
    ssim_values = [m['ssim'] for m in frame_metrics]
    plt.plot(ssim_values, color='blue', alpha=0.7)
    avg_ssim = np.mean(ssim_values)
    plt.axhline(y=avg_ssim, color='red', linestyle='--', alpha=0.7)
    plt.text(0.02, 0.98, f'Average SSIM: {avg_ssim:.4f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'SSIM per Frame - {movie_name}')
    plt.xlabel('Frame')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ssim_{movie_name.lower().replace(" ", "_")}.png'))
    plt.close()
    
    print(f"SSIM plot saved for {movie_name}")
    return avg_ssim

def plot_average_ssim_comparison(avg_ssim_noisy, avg_ssim_denoised, output_dir):
    """Create and save bar plot comparing average SSIM values"""
    print("Creating average SSIM comparison plot...")
    
    plt.figure(figsize=(8, 5))
    
    # SSIM bar plot
    comparisons = ['Clean vs. Noisy', 'Clean vs. Denoised']
    averages = [avg_ssim_noisy, avg_ssim_denoised]
    
    bars = plt.bar(comparisons, averages)
    
    # Add average SSIM value on top of each bar
    for bar, avg in zip(bars, averages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    plt.title('Average SSIM Comparison')
    plt.xlabel('Comparison')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_ssim_comparison.png'))
    plt.close()
    
    print("Average SSIM comparison plot saved")

def main():
    print("Starting SSIM calculation...")
    start_time = time.time()
    
    # Create output directory
    output_dir = r'C:\path\to\output_folder' # Change this
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load clean and noisy simulations
    print("Loading clean and noisy simulations...")
    clean_movie = tifffile.imread(r'C:\path\to\clean_simulation.tif') # Change this
    noisy_movie = tifffile.imread(r'C:\path\to\noisy_simulation.tif') # Change this
    
    # Load and convert denoised simulation from int16 to uint8; if needed only
    print("Loading and converting denoised simulation...")
    denoised_movie = tifffile.imread(r'C:\path\to\noisy_simulation_denoised.tif') # Change this
    print(f"Original denoised dtype: {denoised_movie.dtype}")
    
    # Convert int16 to uint8 by scaling the data; if needed only
    denoised_min = denoised_movie.min()
    denoised_max = denoised_movie.max()
    denoised_movie = ((denoised_movie - denoised_min) * (255.0 / (denoised_max - denoised_min))).astype(np.uint8)
    print(f"Converted denoised dtype: {denoised_movie.dtype}")
    
    # Save converted denoised movie
    temp_denoised_path = os.path.join(output_dir, 'temp_denoised_uint8.tif')
    tifffile.imwrite(temp_denoised_path, denoised_movie)
    
    # Remove first and last 10 frames from clean and noisy; if needed only
    clean_movie = clean_movie[10:-10]  # This will give us 2475 frames
    noisy_movie = noisy_movie[10:-10]  # This will give us 2475 frames
    print(f"Trimmed clean and noisy movies to {len(clean_movie)} frames (removed first and last 10 frames)")
    
    # Save trimmed movies temporarily
    temp_clean_path = os.path.join(output_dir, 'temp_clean.tif')
    temp_noisy_path = os.path.join(output_dir, 'temp_noisy.tif')
    tifffile.imwrite(temp_clean_path, clean_movie)
    tifffile.imwrite(temp_noisy_path, noisy_movie)
    
    # Define movie pairs with trimmed files
    movie_pairs = [
        {
            'Clean Simulation': temp_clean_path,
            'Noisy Simulation': temp_noisy_path,
            'Noisy Simulation Denoised': temp_denoised_path
        }
    ]
    
    # Process all pairs and collect metrics
    print("\nProcessing Clean vs. Noisy comparison...")
    noisy_metrics, noisy_frame_metrics = process_ssim_for_movie(
        movie_pairs[0]['Clean Simulation'],
        movie_pairs[0]['Noisy Simulation']
    )
    avg_ssim_noisy = plot_ssim_over_time(noisy_frame_metrics, 'Clean vs Noisy', output_dir)
    
    print("\nProcessing Clean vs. Denoised comparison...")
    denoised_metrics, denoised_frame_metrics = process_ssim_for_movie(
        movie_pairs[0]['Clean Simulation'],
        movie_pairs[0]['Noisy Simulation Denoised']
    )
    avg_ssim_denoised = plot_ssim_over_time(denoised_frame_metrics, 'Clean vs Denoised', output_dir)
    
    # Create comparison plot of averages
    plot_average_ssim_comparison(avg_ssim_noisy, avg_ssim_denoised, output_dir)
    
    # Save metrics to CSV
    print("\nSaving metrics to CSV...")
    metrics_list = [
        {'comparison': 'Clean vs. Noisy', 'average_ssim': avg_ssim_noisy},
        {'comparison': 'Clean vs. Denoised', 'average_ssim': avg_ssim_denoised}
    ]
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, 'ssim_summary.csv'), index=False)
    print(f"Metrics saved to {os.path.join(output_dir, 'ssim_summary.csv')}")
    
    # Clean up temporary files
    os.remove(temp_clean_path)
    os.remove(temp_noisy_path)
    os.remove(temp_denoised_path)
    print("Cleaned up temporary files")
    
    total_time = time.time() - start_time
    print(f"\nAll processing completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 