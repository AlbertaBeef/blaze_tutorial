import numpy as np
import cv2
import argparse
import os
from pathlib import Path

from axelera import compiler
from axelera.compiler import CompilerConfig

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--name'       , type=str,  default="palm_detection_lite", help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-m', '--model'      , type=str,  default="models/palm_detection_lite.onnx", help="Model file (ONNX format). Default is 'models/palm_detection_lite.onnx'")
ap.add_argument('-r', '--resolution' , type=int,  default=192, help="Input resolution.  Default is 192 for 192x192.")
ap.add_argument('-c', '--cores'      , type=int,  default=1, help="Number of AIPU cores (1-4).  Default is 1.")
ap.add_argument('-s', '--scheme'     , type=str,  default="per_tensor_histogram", help="PTQ scheme.  Default is 'per_tensor_histogram'.")


args = ap.parse_args()

print('Command line options:')
print(' --name        : ', args.name)
print(' --model       : ', args.model)
print(' --resolution  : ', args.resolution)
print(' --cores       : ', args.cores)
print(' --scheme      : ', args.scheme)

model_path = args.model
model_name = args.name
input_shape = (1, args.resolution, args.resolution, 3)

#################################
# Compiler Configuration
#################################

print(f"[INFO] Creating Compiler Configuration ...")

try:

    config = CompilerConfig(
        ptq_scheme=args.scheme,
        aipu_cores=args.cores,
        resources=1.0,
    )

    print(f"[SUCCESS] Creating Compiler Configuration !")

except Exception as e:
    print(f"[ERROR] Creating Compiler Configuration ... ({e})")

#################################
# Calibration Data
#################################

print(f"[INFO] Creating Calibration Data ...")

try:

    if model_name == "palm_detection_v0_07":
        # Specify calibration dataset
        calib_dataset_file = "calib_palm_detection_256_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "hand_landmark_v0_07":
        # Specify calibration dataset
        calib_dataset_file = "calib_hand_landmark_256_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "palm_detection_lite" or args.name == "palm_detection_full":
        # Specify calibration dataset
        calib_dataset_file = "calib_palm_detection_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "hand_landmark_lite" or args.name == "hand_landmark_full":
        # Specify calibration dataset
        calib_dataset_file = "calib_hand_landmark_224_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "face_detection_short_range":
        # Specify calibration dataset
        calib_dataset_file = "calib_face_detection_128_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "face_detection_full_range":
        # Specify calibration dataset
        calib_dataset_file = "calib_face_detection_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "face_landmark":
        # Specify calibration dataset
        calib_dataset_file = "calib_face_landmark_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "pose_detection":
        # Specify calibration dataset
        calib_dataset_file = "calib_pose_detection_224_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if model_name == "pose_landmark_lite" or args.name == "pose_landmark_full" or args.name == "pose_landmark_heavy":
        # Specify calibration dataset
        calib_dataset_file = "calib_pose_landmark_256_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    # Randomize calibration dataset
    print("[INFO] calib_dataset_file : ",calib_dataset_file )
    print("[INFO] calib_dataset (before shuffle) :",
        "shape = ", calib_dataset.shape,
        "| dtype = ", calib_dataset.dtype,
        "| range = ", np.min(np.min(calib_dataset)), "-", np.max(np.max(calib_dataset))
        )
    calib_dataset = np.take(calib_dataset,np.random.permutation(calib_dataset.shape[0]),axis=0,out=calib_dataset)
    print("[INFO] calib_dataset (after  shuffle) :",
        "shape = ", calib_dataset.shape,
        "| dtype = ", calib_dataset.dtype,
        "| range = ", np.min(np.min(calib_dataset)), "-", np.max(np.max(calib_dataset))
        )

    #
    # Create Calibration Data Iterator
    # The Voyager SDK compiler.quantize() function requires an iterator
    # that yields numpy arrays.
    # Each yielded array should be a single preprocessed image:
    #    (1, H, W, 3) float32, range [0.0, 1.0]
    #

    nb_images = calib_dataset.shape[0]
    def calibration_iterator():
        for i in range(nb_images):
            image = calib_dataset[i,:,:,:]
            # image => (H,W,3) uint8 0-255
            sample_input = np.array(image).astype(np.float32) / 255.0
            # sample_input => (H,W,3) float32 0.0-1.0
            sample_input = np.expand_dims(sample_input, 0)
            # sample_input => (1,H,W,3) float32 0.0-1.0
            yield sample_input

    print("[INFO] calibration_iterator :",
        nb_images, "images,",
        "shape =", input_shape
        )

    print(f"[SUCCESS] Creating Calibration Data !")

except Exception as e:
    print(f"[ERROR] Creating Calibration Data ... ({e})")

#################################
# Model Quantization
#################################

print(f"[INFO] Quantizing Model ...")

try:

    #
    # Quantize Model
    # References :
    #    https://github.com/axelera-ai-hub/voyager-sdk
    #    compiler.quantize() takes an ONNX model, calibration data iterator, and config.
    #    Returns a quantized model object (AxeleraQuantizedModel).
    #

    quantized_model = compiler.quantize(
        model=model_path,
        calibration_dataset=calibration_iterator(),
        config=config,
    )

    print(f"[SUCCESS] Quantizing Model !")

except Exception as e:
    print(f"[ERROR] Quantizing Model ... ({e})")

#################################
# Model Compilation
#################################

print(f"[INFO] Compiling Model ...")

try:

    #
    # Compile Model
    # References :
    #    https://github.com/axelera-ai-hub/voyager-sdk
    #    compiler.compile() takes the quantized model, config, and output directory.
    #    Produces a directory with model.json + binaries for the Metis AIPU.
    #

    output_dir = Path(f"./compiled_{model_name}")

    compiler.compile(
        model=quantized_model,
        config=config,
        output_dir=output_dir,
    )

    print(f"[SUCCESS] Compiling Model ! Output directory: {output_dir}")

except Exception as e:
    print(f"[ERROR] Compiling Model ... ({e})")
