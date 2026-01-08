import numpy as np
import cv2
import argparse
import os

import qai_hub as hub

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch'       , type=str,  default="Dragonwing RB3 Gen 2 Vision Kit", help="Qualcomm Target Device.  Default is 'Dragonwing RB3 Gen 2 Vision Kit'.")
ap.add_argument('-b', '--blaze'      , type=str,  default="hand", help="Blaze application ('hand', 'face', 'pose').  Default is 'hand'")
ap.add_argument('-n', '--name'       , type=str,  default="palm_detection_lite", help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-m', '--model'      , type=str,  default="models/palm_detection_lite.onnx", help="Model file (ONNX format). Default is 'models/palm_detection_lite.onnx'")
ap.add_argument('-r', '--resolution' , type=int,  default=192, help="Input resolution.  Default is 192 for 192x192.")


args = ap.parse_args()  
  
print('Command line options:')
print(' --arch        : ', args.arch)
print(' --blaze       : ', args.blaze)
print(' --name        : ', args.name)
print(' --model       : ', args.model)
print(' --resolution  : ', args.resolution)

model_path = args.model
model_name = args.name
device = device = hub.Device(args.arch)
input_shape = (1, args.resolution, args.resolution, 3)

# Create Calibration Data
# Reference : https://aihub.qualcomm.com/get-started#workbench (section 7.b)

if True:
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

    nb_images = calib_dataset.shape[0]
    sample_inputs = []
    for i in range(nb_images):
        image = calib_dataset[i,:,:,:]
        # image => (192,192,3) uint8 0-255
        sample_input = np.array(image).astype(np.float32) / 255.0
        # sample_input => (192,192,3) float32 0.0-1.0
        #sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
        # sample_input => (1,3,192,192) float32 0.0-1.0
        sample_input = np.expand_dims(sample_input, 0)
        # sample_input => (1,192,192,3) float32 0.0-1.0
        sample_inputs.append(sample_input.astype(np.float32))
    #print("[INFO] sample_inputs : ",sample_inputs.shape, sample_inputs.dtype, np.min(np.min(sample_inputs)), np.max(np.max(sample_inputs)) )        
    calibration_data = {"input": sample_inputs}        
    print("[INFO] calibration_data (formatted for QAI HUB Workbench) :",
        "shape = ", len(calibration_data["input"]), "x", calibration_data["input"][0].shape, 
        "| dtype = ", calibration_data["input"][0].dtype,
        "| range = ", np.min(np.min(calibration_data["input"])), "-", np.max(np.max(calibration_data["input"])) 
        )


#
# Submit Compile Job
# Reference : https://aihub.qualcomm.com/get-started#workbench (sections 3 & 7.a)
#

# Submit compile job for the TFLite model
compile_job = hub.submit_compile_job(
    model=model_path,
    device=device,
    input_specs={"input": input_shape},
    options="--target_runtime onnx"
)

# Wait for completion and fetch the target-optimized model
unquantized_onnx_model = compile_job.get_target_model()

#
# Submit Profile Job
# Reference : https://aihub.qualcomm.com/get-started#workbench (section 6)
#

# Submit profile job
profile_job = hub.submit_profile_job(
    model=unquantized_onnx_model,
    device=device,
)


#
# Submit Quantization Job
# Reference : https://aihub.qualcomm.com/get-started#workbench
#

# Submit quantize job
quantize_job = hub.submit_quantize_job(
    model=unquantized_onnx_model,
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)

# Wait for completion and fetch the quantized model
quantized_onnx_model = quantize_job.get_target_model()

#
# Submit Optimize Job
# Reference : https://aihub.qualcomm.com/get-started#workbench (section 7.c)
#

# Optimize model for the chosen device
compile_job = hub.submit_compile_job(
        model=quantized_onnx_model,
        device=device,
        input_specs={"input": input_shape},
        options="--target_runtime tflite"
)
target_model = compile_job.get_target_model()

target_model.download(model_name+".tflite")

#
# Submit Profile Job
# Reference : https://aihub.qualcomm.com/get-started#workbench (section 6)
#

# Submit profile job
profile_job = hub.submit_profile_job(
    model=target_model,
    device=device,
)
