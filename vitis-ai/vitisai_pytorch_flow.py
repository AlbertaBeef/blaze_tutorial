import cv2
import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)

sys.path.append(os.path.abspath('../MediaPipePyTorch/'))

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark
from blazepose import BlazePose
from blazepose_landmark import BlazePoseLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS, POSE_CONNECTIONS



from pytorch_nndct.apis import Inspector
from pytorch_nndct.apis import torch_quantizer, dump_xmodel


# use GPU if available   
if (torch.cuda.device_count() > 0):
  print('You have',torch.cuda.device_count(),'CUDA devices available')
  for i in range(torch.cuda.device_count()):
    print(' Device',str(i),': ',torch.cuda.get_device_name(i))
  print('Selecting device 0..')
  device = torch.device('cuda:0')
else:
  print('No CUDA devices available..selecting CPU')
  device = torch.device('cpu')

 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch'       , type=str,  default="", help="Hailo HW architecture.  Default is 'hailo8'.")
ap.add_argument('-n', '--name'       , type=str,  default="palm_detection_lite", help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-r', '--resolution' , type=int,  default=192, help="Input resolution.  Default is 192 for 192x192.")
ap.add_argument('-p', '--process'    , type=str,  default="all", help="Command seperated list of processes to run ( 'inspect', 'quantize', 'all'=='parse,calibrate,test' ). Default is 'all'")

args = ap.parse_args()  
  
print('Command line options:')
print(' --arch        : ', args.arch)
print(' --name        : ', args.name)
print(' --resolution  : ', args.resolution)
print(' --process     : ', args.process)

if args.name == "BlazePalm":
    model = BlazePalm().to(device)
    model.load_weights("../MediaPipePyTorch/blazepalm.pth")
    model.load_anchors("../MediaPipePyTorch/anchors_palm.npy")
    model.min_score_thresh = .75
    
    calib_dataset = np.load("calib_palm_detection_256_dataset.npy")
    
elif args.name == "BlazeHandLandmark":
    model = BlazeHandLandmark().to(device)
    model.load_weights("../MediaPipePyTorch/blazehand_landmark.pth")
    
    calib_dataset = np.load("calib_hand_landmark_256_dataset.npy")
    
elif args.name == "BlazeFace":
    model = BlazeFace(back_model=False).to(device)
    model.load_weights("../MediaPipePyTorch/blazeface.pth")
    model.load_anchors("../MediaPipePyTorch/anchors_face.npy")

    calib_dataset = np.load("calib_face_detection_128_dataset.npy")
    
elif args.name == "BlazeFaceBack":
    model = BlazeFace(back_model=True).to(device)
    model.load_weights("../MediaPipePyTorch/blazefaceback.pth")
    model.load_anchors("../MediaPipePyTorch/anchors_face_back.npy")

    calib_dataset = np.load("calib_face_detection_256_dataset.npy")
    
elif args.name == "BlazeFaceLandmark":
    model = BlazeFaceLandmark().to(device)
    model.load_weights("../MediaPipePyTorch/blazeface_landmark.pth")

    calib_dataset = np.load("calib_face_landmark_192_dataset.npy")
    
elif args.name == "BlazePose":
    model = BlazePose().to(device)
    model.load_weights("../MediaPipePyTorch/blazepose.pth")
    model.load_anchors("../MediaPipePyTorch/anchors_pose.npy")

    calib_dataset = np.load("calib_pose_detection_128_dataset.npy")
    
elif args.name == "BlazePoseLandmark":
    model = BlazePoseLandmark().to(device)
    model.load_weights("../MediaPipePyTorch/blazepose_landmark.pth")

    calib_dataset = np.load("calib_pose_landmark_256_dataset.npy")

model.eval() 

#print("[INFO] model.summary = ",model.summary())

calib_dataset = np.take(calib_dataset,np.random.permutation(calib_dataset.shape[0]),axis=0,out=calib_dataset)
print("[INFO] calib_dataset shape = ",calib_dataset.shape,calib_dataset.dtype,np.amin(calib_dataset),np.amax(calib_dataset))    
calib_dataset = calib_dataset.astype(np.float32)/256.0
print("[INFO] calib_dataset shape = ",calib_dataset.shape,calib_dataset.dtype,np.amin(calib_dataset),np.amax(calib_dataset))    
calib_dataset_size = calib_dataset.shape[0]
print("[INFO] calib_dataset size = ",calib_dataset_size)    
batch_size = min(1000,calib_dataset_size)
print("[INFO] batch size = ",batch_size)    


#
# Inspect
# Reference : 
#

if ("inspect" in args.process):

    print("[INFO] Model Inspection")

    inspector = Inspector("DPUCZDX8G_ISA1_B4096")
    batchsize = 1
    #rand_in = torch.randn([batchsize, 3,256,256])
    #inspector.inspect(model, (rand_in), device=device)
    #model_input = torch.from_numpy(calib_dataset).permute((0, 3, 1, 2))
    model_input = torch.from_numpy(calib_dataset[0,:,:,:]).permute((2, 0, 1))
    model_input = model_input.unsqueeze(0)
    print("[INFO] model_input shape = ",model_input.shape,model_input.dtype)    

    inspector.inspect(model, (model_input), device=device)    

#
# Quantization
# Reference : 
#

if ("calibrate" in args.process or "quantize" in args.process or args.process == "all"):

    print("[INFO] Quantization (Calibration Phase)")

    quant_mode = 'calib'
    batchsize = batch_size
    quant_model = './model_quant_'+args.name
    
    # force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # override batchsize if in test mode
    if (quant_mode=='test'):
      batchsize = 1
  
    #rand_in = torch.randn([batchsize, 3,256,256])
    model_input = torch.from_numpy(calib_dataset[0:batchsize,:,:,:]).permute((0, 3, 1, 2))
    print("[INFO] model_input shape = ",model_input.shape,model_input.dtype)    
    quantizer = torch_quantizer(quant_mode, model, (model_input), output_dir=quant_model, device=device) 
    quantized_model = quantizer.quant_model

    #acc1_gen, acc5_gen, loss_gen = evaluate(quantized_model, val_loader, loss_fn)

    # reference : https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py
    quantized_model.eval()
    quantized_model = quantized_model.to(device)
    batchsize=1
    #inputs = torch.randn([batchsize, 3,256,256])
    model_input = torch.from_numpy(calib_dataset[0:batchsize,:,:,:]).permute((0, 3, 1, 2))
    print("[INFO] model_input shape = ",model_input.shape,model_input.dtype)    
    model_outputs = quantized_model(model_input)
    print("[INFO] model_outputs shape = ",len(model_outputs))    

    if quant_mode == 'calib':
        quantizer.export_quant_config()    
        
if ("test" in args.process or "quantize" in args.process or args.process == "all"):

    print("[INFO] Quantization (Test Phase)")
    quant_mode = 'test'
    batchsize = 1
    quant_model = './model_quant_'+args.name

    #rand_in = torch.randn([batchsize, 3,256,256])
    model_input = torch.from_numpy(calib_dataset[0:batchsize,:,:,:]).permute((0, 3, 1, 2))
    print("[INFO] model_input shape = ",model_input.shape,model_input.dtype)    
    quantizer = torch_quantizer(quant_mode, model, (model_input), output_dir=quant_model, device=device) 
    quantized_model = quantizer.quant_model

    # reference : https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_quantizer/vai_q_pytorch/example/resnet18_quant.py
    quantized_model.eval()
    quantized_model = quantized_model.to(device)
    #inputs = torch.randn([batchsize, 3,256,256])
    model_input = torch.from_numpy(calib_dataset[0:batchsize,:,:,:]).permute((0, 3, 1, 2))
    print("[INFO] model_input shape = ",model_input.shape,model_input.dtype)    
    model_outputs = quantized_model(model_input)
    print("[INFO] model_outputs shape = ",len(model_outputs))    

    quantizer.export_torch_script()
    
    quantizer.export_onnx_model()
    
    quantizer.export_xmodel(deploy_check=False)

