import sys
sys.path.append("C:/path/to/deepinterpolation/folder") # Change this

import tensorflow as tf
import os
import datetime
from deepinterpolation.generator_collection import SingleTifGenerator
from deepinterpolation.generator_collection import CollectorGenerator
from deepinterpolation.trainor_collection import core_trainer
from tensorflow.keras.callbacks import CSVLogger, TensorBoard
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, AveragePooling2D

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def main():
    # Add this at the start of main()
    if os.name == 'nt':  # Windows
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    # ==== GPU Setup ====
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {gpus}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Optionally, limit memory usage to 80% of available GPU memory
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 6)]  # Limit to 6GB
                )
            print(f"Enabled memory growth for GPU: {gpus}")
        except RuntimeError as e:
            print(e)
            
    # Add memory cleanup
    tf.keras.backend.clear_session()

    # ==== Paths ====
    train_files = [
        r"C:\path\to\movie1.tif",
        r"C:\path\to\movie2.tif",
        r"C:\path\to\\movie3.tif",
        # Add/remove files as needed
    ]
    valid_file = r"C:\path\to\movie4.tif" # Change this
    output_dir = r"C:\path\to\trained_models" # Change this

    # ==== Run ID ====
    run_uid = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # ==== Model name ====
    model_name = f"condition2_unet_single_flexible_{run_uid}"

    # ==== Generator Parameters ====
    generator_param = {
        "steps_per_epoch": 5,
        "pre_post_frame":10,
        "batch_size": 1,
        "start_frame": 0,
        "end_frame": 200,
        "pre_post_omission": 0,
    }

    generator_test_param = {
        "steps_per_epoch": 2,
        "pre_post_frame": 10,
        "train_path": valid_file,
        "batch_size": 1,
        "start_frame": 0,
        "end_frame": 100,
        "pre_post_omission": 0,
    }

    # ==== Training Parameters ====
    training_param = {
        "run_uid": run_uid,
        "output_dir": os.path.join(output_dir, model_name),
        "model_string": model_name,
        "loss": "mean_absolute_error",
        "nb_gpus": 1,
        "apply_learning_decay": 0,
        "nb_times_through_data": 1,
        "batch_size": 1,
        "steps_per_epoch": 5,
        "period_save": 10,
        "learning_rate": 1e-4,
        "pre_post_frame": 10,
        "nb_workers": 0,
        "use_multiprocessing": False
    }

    os.makedirs(training_param["output_dir"], exist_ok=True)

    # ==== Generators ====
    # Create multiple training generators
    train_generators = []
    for train_file in train_files:
        gen_params = generator_param.copy()
        gen_params["train_path"] = train_file
        train_generators.append(SingleTifGenerator(gen_params))
    
    # Combine them using CollectorGenerator
    train_generator = CollectorGenerator(train_generators)
    test_generator = SingleTifGenerator(generator_test_param)

    # ==== Network ====
    # Create a super lightweight network
    def create_lightweight_unet(input_shape):
        inputs = Input(shape=input_shape)
        
        # Initial downsampling to reduce memory usage
        x = AveragePooling2D(2)(inputs)  # Reduce spatial dimensions by 2x
        
        # Encoder - minimal filters
        conv1 = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(16, (3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bridge
        conv3 = Conv2D(32, (3, 3), activation="relu", padding="same")(pool2)

        # Decoder - minimal filters
        up1 = UpSampling2D(size=(2, 2))(conv3)
        merge1 = Concatenate()([up1, conv2])
        conv4 = Conv2D(16, (3, 3), activation="relu", padding="same")(merge1)

        up2 = UpSampling2D(size=(2, 2))(conv4)
        merge2 = Concatenate()([up2, conv1])
        conv5 = Conv2D(8, (3, 3), activation="relu", padding="same")(merge2)

        # Final upsampling to original size
        outputs = UpSampling2D(2)(conv5)
        outputs = Conv2D(1, (1, 1), activation=None, padding="same")(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    # ==== Create and compile model ====
    input_size = train_generator.get_input_size()
    model = create_lightweight_unet(input_size)
    model.summary()  # Print model summary to verify the reduced size

    # ==== Logging ====
    log_dir = os.path.join(training_param["output_dir"], "logs")
    csv_logger = CSVLogger(os.path.join(training_param["output_dir"], "training_log.csv"))
    tensorboard_logger = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ==== Trainer ====
    trainer = core_trainer(train_generator, test_generator, model, training_param)
    trainer.callbacks = [csv_logger, tensorboard_logger]

    # Add debug prints before training
    print(f"Training generator size: {len(train_generator)}")
    print(f"Test generator size: {len(test_generator)}")
    print(f"Training parameters: {training_param}")

    # Add checks for data files
    assert os.path.exists(train_file), f"Training file not found: {train_file}"
    assert os.path.exists(valid_file), f"Validation file not found: {valid_file}"

    # ==== Run Training ====
    print(f"Starting training: {model_name}")
    trainer.run()
    trainer.finalize()
    print("Training complete.")

if __name__ == "__main__":
    main()
