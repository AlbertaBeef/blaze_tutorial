# Copyright 2024 Avnet Inc.
# Licensed under the Apache License, Version 2.0
#
# DeepX M1 compilation flow for MediaPipe Blaze models
#
# Generates a DX-COM JSON configuration and compiles an ONNX model
# to a .dxnn binary for the DeepX M1 NPU.
#
# Prerequisites:
#   1. DX-COM compiler (dx_com) must be installed and on PATH
#      - See deepx/DeepX_Software_Tooling.md for installation instructions
#      - Verify with: dx_com -v
#
#   2. ONNX models must be prepared:
#      cd deepx/models
#      bash get_tflite_models.sh     # Download TFLite models
#      bash convert_models.sh        # Convert TFLite to ONNX via tf2onnx
#
#   3. Calibration datasets (.npy files) must be in the deepx/ directory:
#      - calib_palm_detection_256_dataset.npy
#      - calib_hand_landmark_256_dataset.npy
#      - calib_palm_detection_192_dataset.npy
#      - calib_hand_landmark_224_dataset.npy
#      - calib_face_detection_128_dataset.npy
#      - calib_face_detection_192_dataset.npy
#      - calib_face_landmark_192_dataset.npy
#      - calib_pose_detection_224_dataset.npy
#      - calib_pose_landmark_256_dataset.npy
#
#   4. Python packages: numpy (for calibration data verification)
#      Optional: onnx (for auto-detecting ONNX input tensor names)
#
# Usage:
#   python3 deepx_flow.py --name palm_detection_lite --model models/palm_detection_lite.onnx --resolution 192
#   python3 deepx_flow.py --name palm_detection_lite --model models/palm_detection_lite.onnx --resolution 192 --calib_method ema --shrink

import numpy as np
import json
import subprocess
import argparse
import os
import sys

#################################
# Model Registry
#################################
# Maps model names to their ONNX input tensor names (from TFLite),
# default resolutions, and calibration dataset filenames.
#
# ONNX input names are derived from the TFLite input names:
#   - Most models use "input_1"
#   - face_detection variants use "input"
#
# Note: tf2onnx may rename inputs. If auto-detection is available
# (via the onnx package), it will override these defaults.

MODEL_REGISTRY = {
    # v0.07 models
    "palm_detection_v0_07": {
        "input_name": "input_1",
        "resolution": 256,
        "calib_dataset": "calib_palm_detection_256_dataset.npy",
    },
    "hand_landmark_v0_07": {
        "input_name": "input_1",
        "resolution": 256,
        "calib_dataset": "calib_hand_landmark_256_dataset.npy",
    },
    # v0.10 hand models
    "palm_detection_lite": {
        "input_name": "input_1",
        "resolution": 192,
        "calib_dataset": "calib_palm_detection_192_dataset.npy",
    },
    "palm_detection_full": {
        "input_name": "input_1",
        "resolution": 192,
        "calib_dataset": "calib_palm_detection_192_dataset.npy",
    },
    "hand_landmark_lite": {
        "input_name": "input_1",
        "resolution": 224,
        "calib_dataset": "calib_hand_landmark_224_dataset.npy",
    },
    "hand_landmark_full": {
        "input_name": "input_1",
        "resolution": 224,
        "calib_dataset": "calib_hand_landmark_224_dataset.npy",
    },
    # v0.10 face models
    "face_detection_short_range": {
        "input_name": "input",
        "resolution": 128,
        "calib_dataset": "calib_face_detection_128_dataset.npy",
    },
    "face_detection_full_range": {
        "input_name": "input",
        "resolution": 192,
        "calib_dataset": "calib_face_detection_192_dataset.npy",
    },
    "face_landmark": {
        "input_name": "input_1",
        "resolution": 192,
        "calib_dataset": "calib_face_landmark_192_dataset.npy",
    },
    # v0.10 pose models
    "pose_detection": {
        "input_name": "input_1",
        "resolution": 224,
        "calib_dataset": "calib_pose_detection_224_dataset.npy",
    },
    "pose_landmark_lite": {
        "input_name": "input_1",
        "resolution": 256,
        "calib_dataset": "calib_pose_landmark_256_dataset.npy",
    },
    "pose_landmark_full": {
        "input_name": "input_1",
        "resolution": 256,
        "calib_dataset": "calib_pose_landmark_256_dataset.npy",
    },
    "pose_landmark_heavy": {
        "input_name": "input_1",
        "resolution": 256,
        "calib_dataset": "calib_pose_landmark_256_dataset.npy",
    },
}

#################################
# Argument Parser
#################################

ap = argparse.ArgumentParser(
    description="Compile ONNX models for the DeepX M1 NPU using DX-COM."
)
ap.add_argument('-n', '--name',         type=str, default="palm_detection_lite",
                help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-m', '--model',        type=str, default="models/palm_detection_lite.onnx",
                help="Model file (ONNX format). Default is 'models/palm_detection_lite.onnx'")
ap.add_argument('-r', '--resolution',   type=int, default=192,
                help="Input resolution. Default is 192 for 192x192.")
ap.add_argument('-c', '--calib_method', type=str, default="minmax", choices=["minmax", "ema"],
                help="Calibration method. Default is 'minmax'.")
ap.add_argument('-k', '--calib_num',    type=int, default=100,
                help="Number of calibration steps. Default is 100.")
ap.add_argument('-s', '--shrink',       action='store_true',
                help="Use --shrink flag for minimal deterministic output.")
ap.add_argument('-o', '--output',       type=str, default=None,
                help="Output directory. Default is 'compiled_<name>'.")

args = ap.parse_args()

print('Command line options:')
print(' --name         : ', args.name)
print(' --model        : ', args.model)
print(' --resolution   : ', args.resolution)
print(' --calib_method : ', args.calib_method)
print(' --calib_num    : ', args.calib_num)
print(' --shrink       : ', args.shrink)
print(' --output       : ', args.output)

model_name = args.name
model_path = args.model
resolution = args.resolution

#################################
# Model Configuration Lookup
#################################

print(f"[INFO] Looking up model configuration for '{model_name}' ...")

if model_name in MODEL_REGISTRY:
    model_info = MODEL_REGISTRY[model_name]
    input_name = model_info["input_name"]
    calib_dataset_file = model_info["calib_dataset"]
    expected_resolution = model_info["resolution"]
    if resolution != expected_resolution:
        print(f"[WARNING] Resolution {resolution} does not match expected {expected_resolution} for {model_name}")
else:
    print(f"[WARNING] Model '{model_name}' not found in registry. Using defaults.")
    input_name = "input_1"
    calib_dataset_file = f"calib_{model_name}_{resolution}_dataset.npy"

# DX-COM expects NCHW input shape (ONNX convention)
input_shape = [1, 3, resolution, resolution]

print(f"[INFO] Input name     : {input_name}")
print(f"[INFO] Input shape    : {input_shape}")
print(f"[INFO] Calib dataset  : {calib_dataset_file}")

#################################
# ONNX Input Name Auto-Detection
#################################
# If the onnx package is installed, verify the input tensor name
# from the actual ONNX model file.

try:
    import onnx
    if os.path.isfile(model_path):
        onnx_model = onnx.load(model_path)
        detected_name = onnx_model.graph.input[0].name
        if detected_name != input_name:
            print(f"[WARNING] ONNX model input name '{detected_name}' differs from registry '{input_name}'")
            print(f"[INFO] Using detected name: '{detected_name}'")
            input_name = detected_name
        else:
            print(f"[INFO] ONNX input name verified: '{detected_name}'")
except ImportError:
    print(f"[INFO] onnx package not installed, using registry input name")
except Exception as e:
    print(f"[WARNING] Could not auto-detect ONNX input name: {e}")

#################################
# Verify Prerequisites
#################################

print(f"[INFO] Verifying prerequisites ...")

# Check ONNX model exists
if not os.path.isfile(model_path):
    print(f"[ERROR] ONNX model not found: {model_path}")
    print(f"[INFO]  Run 'bash models/get_tflite_models.sh' and 'cd models && bash convert_models.sh' first.")
    sys.exit(1)

# Check calibration dataset exists
if not os.path.isfile(calib_dataset_file):
    print(f"[ERROR] Calibration dataset not found: {calib_dataset_file}")
    print(f"[INFO]  Place calibration .npy files in the deepx/ directory.")
    sys.exit(1)

# Verify calibration dataset shape
calib_dataset = np.load(calib_dataset_file)
print(f"[INFO] Calib dataset  : shape={calib_dataset.shape}, dtype={calib_dataset.dtype}, "
      f"range={np.min(calib_dataset)}-{np.max(calib_dataset)}")
expected_shape_suffix = (resolution, resolution, 3)
if calib_dataset.shape[1:] != expected_shape_suffix:
    print(f"[WARNING] Calib dataset shape {calib_dataset.shape} does not match "
          f"expected (N, {resolution}, {resolution}, 3)")
del calib_dataset  # free memory

# Check dx_com is available
try:
    result = subprocess.run(["dx_com", "-v"], capture_output=True, text=True, timeout=10)
    print(f"[INFO] dx_com version: {result.stdout.strip()}")
except FileNotFoundError:
    print(f"[ERROR] dx_com not found on PATH.")
    print(f"[INFO]  Install the DeepX DX-COM compiler.")
    print(f"[INFO]  See deepx/DeepX_Software_Tooling.md for installation instructions.")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print(f"[WARNING] dx_com -v timed out, proceeding anyway ...")
except Exception as e:
    print(f"[WARNING] Could not verify dx_com: {e}")

print(f"[SUCCESS] Prerequisites verified !")

#################################
# Generate JSON Configuration
#################################

print(f"[INFO] Generating JSON configuration ...")

config = {
    "inputs": {
        input_name: input_shape
    },
    "calibration_method": args.calib_method,
    "calibration_num": args.calib_num,
    "default_loader": {
        "dataset_path": calib_dataset_file,
        "file_extensions": [".npy"],
        "preprocessing": {}
    }
}

# Create configs directory if needed
os.makedirs("configs", exist_ok=True)
config_path = f"configs/{model_name}.json"

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"[INFO] Config saved to: {config_path}")
print(json.dumps(config, indent=4))
print(f"[SUCCESS] JSON configuration generated !")

#################################
# Compile with DX-COM
#################################

print(f"[INFO] Compiling model with DX-COM ...")

output_dir = args.output if args.output else f"compiled_{model_name}"

cmd = [
    "dx_com",
    "-m", model_path,
    "-c", config_path,
    "-o", output_dir,
]
if args.shrink:
    cmd.append("--shrink")

print(f"[INFO] Command: {' '.join(cmd)}")

try:
    result = subprocess.run(cmd, text=True)
    if result.returncode == 0:
        print(f"[SUCCESS] Model compiled ! Output directory: {output_dir}")
    else:
        print(f"[ERROR] dx_com returned exit code {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Compilation failed: {e}")
    sys.exit(1)
