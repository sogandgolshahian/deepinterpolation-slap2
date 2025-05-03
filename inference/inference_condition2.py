import sys
sys.path.append("C:/path/to/deepinterpolation/folder") # Change this

import numpy as np
import tensorflow as tf
import h5py
import tifffile
from deepinterpolation.generator_collection import SingleTifGenerator
from deepinterpolation.inference_collection import core_inference

# Optional temporal average function for improved inference results
def temporal_average(input_path, window_size):
    """
    Load and average the movie temporally using a sliding window
    """
    # Load the movie
    movie = tifffile.imread(input_path)
    
    # Create averaged movie
    n_frames = movie.shape[0]
    averaged_movie = np.zeros_like(movie)
    
    # Sliding window average
    for i in range(n_frames):
        start_idx = max(0, i - window_size//2)
        end_idx = min(n_frames, i + window_size//2 + 1)
        averaged_movie[i] = np.mean(movie[start_idx:end_idx], axis=0)
    
    # Save averaged movie
    output_path = input_path.replace('.tif', '_averaged.tif')
    tifffile.imwrite(output_path, averaged_movie)
    return output_path

def main():
    # ==== GPU Setup ====
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # ==== Paths ====
    inference_file = r"C:\path\to\inference_file.tif" # Change this
    # averaged_inference_file = temporal_average(inference_file, window_size = 3)
    model_path = r"C:\path\to\model.h5" # Change this
    output_h5_path = r"C:\path\to\output_file.h5" # Change this
    output_tif_path = output_h5_path.replace(".h5", ".tif")

    # ==== Generator parameters ====
    generator_param = {
        "train_path": inference_file, # Can replace this with averaged_inference_file
        "pre_post_frame": 10,
        "steps_per_epoch": -1,
        "batch_size": 1,
        "start_frame": 0,
        "end_frame": -1,
        "pre_post_omission": 0,
        "randomize": 0
    }

    # ==== Inference parameters ====
    inference_param = {
        "model_path": model_path,
        "output_file": output_h5_path,
        "output_datatype": "uint16" # Can change the output datatype
    }

    # ==== Run Inference ====
    print("\n--- Initializing generator and inference ---")
    generator_obj = SingleTifGenerator(generator_param)
    inference_runner = core_inference(inference_param, generator_obj)

    print("--- Running inference ---")
    inference_runner.run()
    print("\nInference complete. Output saved as HDF5 at:", output_h5_path)

    # ==== Convert to TIFF ====
    print("\n--- Converting HDF5 output to TIFF ---")
    with h5py.File(output_h5_path, "r") as f:
        data = f["data"][()]
        tifffile.imwrite(output_tif_path, data.astype("uint16")) # Change this datatype to match the "output_datatype" above

    print("TIFF saved at:", output_tif_path)

if __name__ == "__main__":
    main()
