'''
Copyright 2024 Avnet Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
#
# Palm Detection (live with USB camera)
#
# References:
#   https://www.github.com/AlbertaBeef/blazepalm_tutorial/tree/2023.1
#   https://github.com/Xilinx/Vitis-AI-Tutorials/tree/3.0/Tutorials/pytorch-subgraphs/#4-dpu-and-cpu-subgraphs-of-the-cnn
#
# Dependencies:
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

#import keras
#from keras.models import load_model
#from keras.utils import to_categorical

from ctypes import *
from typing import List
import xir
import pathlib
import vart
import vitis_ai_library
#import threading
import time
import sys
import argparse
import glob
import subprocess
import re

def get_media_dev_by_name(src):
    devices = glob.glob("/dev/media*")
    for dev in sorted(devices):
        proc = subprocess.run(['media-ctl','-d',dev,'-p'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

def get_video_dev_by_name(src):
    devices = glob.glob("/dev/video*")
    for dev in sorted(devices):
        proc = subprocess.run(['v4l2-ctl','-d',dev,'-D'], capture_output=True, encoding='utf8')
        for line in proc.stdout.splitlines():
            if src in line:
                return dev

# ...work in progress ...
#def detect_dpu_architecture():
#    proc = subprocess.run(['xdputil','query'], capture_output=True, encoding='utf8')
#    for line in proc.stdout.splitlines():
#        if 'DPU Arch' in line:
#            #                 "DPU Arch":"DPUCZDX8G_ISA0_B128_01000020E2012208",
#            #dpu_arch = re.search('DPUCZDX8G_ISA0_(.+?)_', line).group(1)  
#            #                 "DPU Arch":"DPUCZDX8G_ISA1_B2304",
#            #dpu_arch = re.search('DPUCZDX8G_ISA1_(.+?)', line).group(1)
#            return dpu_arch

# Parameters (tweaked for video)
scale = 1.0
text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA

print("[INFO] Searching for USB camera ...")
dev_video = get_video_dev_by_name("uvcvideo")
dev_media = get_media_dev_by_name("uvcvideo")
print(dev_video)
print(dev_media)

#input_video = 0 
input_video = dev_video  
print("[INFO] Input Video : ",input_video)

output_dir = './captured-images'

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist

#cv2.namedWindow('ASL Classification')
cv2.namedWindow('Palm Detection')


# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("camera",input_video," (",frame_width,",",frame_height,")")

# Open ASL model
#model = load_model('tf2_asl_classifier_1.h5')

# Vitis-AI implementation

def get_subgraph (g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    return sub

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]



"""
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
"""
import math
def CPUCalcSoftmax(data, size):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result

"""
Get topk results according to its probability
datain: data result of softmax
filePath: filePath in witch that records the infotmation of kinds
"""

def TopK(datain, size, filePath):

    cnt = [i for i in range(size)]
    pair = zip(datain, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)
    fp = open(filePath, "r")
    data1 = fp.readlines()
    fp.close()
    for i in range(5):
        idx = 0
        for line in data1:
            if idx == cnt_new[i]:
                print("Top[%d] %d %s" % (i, idx, (line.strip)("\n")))
            idx = idx + 1

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()  
ap.add_argument('-m', '--model',     type=str, default='palm_detector.xmodel', help='Path of xmodel. Default is palm_detector.xmodel')

args = ap.parse_args()  
  
print ('Command line options:')
print (' --model     : ', args.model)

#dpu_arch = detect_dpu_architecture()
#print('[INFO] Detected DPU architecture : ',dpu_arch)
#
#model_path = './model_1/'+dpu_arch+'/asl_classifier_1.xmodel'
#print('[INFO] ASL model : ',model_path)
model_path = args.model

# Create graph runner
#g = xir.Graph.deserialize(model_path)
#runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
#input_tensor_buffers = runner.get_inputs()
#output_tensor_buffers = runner.get_outputs()
#print("Input Tensors:")
#print(len(input_tensor_buffers))
#print(input_tensor_buffers)
#print("Output Tensors:")
#print(len(output_tensor_buffers))
#print(output_tensor_buffers)

def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_output_tensors()
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)


#DEBUG = True
DEBUG = False
def DEBUG_runDPU(dpu_1, dpu_3, dpu_5, dpu_7, inputData, outputData):
    if DEBUG:
        print("Start DPU DEBUG with 1 input image")
        
    # get DPU input/output tensors
    inputTensor_1  = dpu_1.get_input_tensors()
    outputTensor_1 = dpu_1.get_output_tensors()
    inputTensor_3  = dpu_3.get_input_tensors()
    outputTensor_3 = dpu_3.get_output_tensors()
    inputTensor_5  = dpu_5.get_input_tensors()
    outputTensor_5 = dpu_5.get_output_tensors()
    inputTensor_7  = dpu_7.get_input_tensors()
    outputTensor_7 = dpu_7.get_output_tensors()
    #if DEBUG:
    if False:
# inputTensor1
# [<xir.Tensor named 'BlazePalm__input_0_fix'>]
#outputTensor1
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__MaxPool2d_max_pool__13016_fix'>]
# inputTensor3
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__ret_51_fix'>]
#outputTensor3
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__MaxPool2d_max_pool__13404_fix'>]
# inputTensor5
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__ret_103_fix'>]
#outputTensor5
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__MaxPool2d_max_pool__13792_fix'>]
# inputTensor7
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__ret_155_fix'>]
#outputTensor7
# [<xir.Tensor named 'BlazePalm__BlazePalm_ret_293_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_ret_fix'>]
    
        print(" inputTensor1\n", inputTensor_1)
        print("outputTensor1\n", outputTensor_1)
        print(" inputTensor3\n", inputTensor_3)
        print("outputTensor3\n", outputTensor_3)
        print(" inputTensor5\n", inputTensor_5)
        print("outputTensor5\n", outputTensor_5)
        print(" inputTensor7\n", inputTensor_7)
        print("outputTensor7\n", outputTensor_7)

    input_1_ndim  = tuple(inputTensor_1[0].dims)
    input_3a_ndim  = tuple(inputTensor_3[0].dims)
    input_3b_ndim  = tuple(inputTensor_3[1].dims)
    input_5a_ndim  = tuple(inputTensor_5[0].dims)
    input_5b_ndim  = tuple(inputTensor_5[1].dims)
    input_7a_ndim  = tuple(inputTensor_7[0].dims)
    input_7b_ndim  = tuple(inputTensor_7[1].dims)
    output_1a_ndim = tuple(outputTensor_1[0].dims)
    output_1b_ndim = tuple(outputTensor_1[1].dims)
    output_3a_ndim = tuple(outputTensor_3[0].dims)
    output_3b_ndim = tuple(outputTensor_3[1].dims)
    output_5a_ndim = tuple(outputTensor_5[0].dims)
    output_5b_ndim = tuple(outputTensor_5[1].dims)
    output_7a_ndim = tuple(outputTensor_7[0].dims)
    output_7b_ndim = tuple(outputTensor_7[1].dims)
    batchSize = input_1_ndim[0]
    
    #batchSize = 1
    #input_1_ndim = (batchSize,) + input_1_ndim[1:]
    #...
    #output_7a_ndim = (batchSize,) + output_7a_ndim[1:]
    #output_7b_ndim = (batchSize,) + output_7b_ndim[1:]

# input_1_ndim:  (1, 256, 256, 3)
#output_1a_ndim:  (1, 128, 128, 32)
#output_1b_ndim:  (1, 64, 64, 32)
# input_3a_ndim:  (1, 128, 128, 32)
# input_3b_ndim:  (1, 64, 64, 64)
#output_3a_ndim:  (1, 64, 64, 64)
#output_3b_ndim:  (1, 32, 32, 64)
# input_5a_ndim:  (1, 64, 64, 64)
# input_5b_ndim:  (1, 32, 32, 128)
#output_5a_ndim:  (1, 32, 32, 128)
#output_5b_ndim:  (1, 16, 16, 128)
# input_7a_ndim:  (1, 32, 32, 128)
# input_7b_ndim:  (1, 16, 16, 256)
#output_7a_ndim:  (1, 2944, 1)
#output_7b_ndim:  (1, 2944, 18)
    #if DEBUG:
    if False:
        print(" input_1_ndim: ",  input_1_ndim)
        print("output_1a_ndim: ", output_1a_ndim)
        print("output_1b_ndim: ", output_1b_ndim)
        print(" input_3a_ndim: ",  input_3a_ndim)
        print(" input_3b_ndim: ",  input_3b_ndim)
        print("output_3a_ndim: ", output_3a_ndim)
        print("output_3b_ndim: ", output_3b_ndim)
        print(" input_5a_ndim: ",  input_5a_ndim)
        print(" input_5b_ndim: ",  input_5b_ndim)
        print("output_5a_ndim: ", output_5a_ndim)
        print("output_5b_ndim: ", output_5b_ndim)
        print(" input_7a_ndim: ",  input_7a_ndim)
        print(" input_7b_ndim: ",  input_7b_ndim)
        print("output_7a_ndim: ", output_7a_ndim)
        print("output_7b_ndim: ", output_7b_ndim)

    out1a = np.zeros(output_1a_ndim, dtype='float32')
    out1b = np.zeros(output_1b_ndim, dtype='float32')
    out3a = np.zeros(output_3a_ndim, dtype='float32')
    out3b = np.zeros(output_3b_ndim, dtype='float32')
    out5a = np.zeros(output_5a_ndim, dtype='float32')
    out5b = np.zeros(output_5b_ndim, dtype='float32')
    out7a = np.zeros(output_7a_ndim, dtype='float32')
    out7b = np.zeros(output_7b_ndim, dtype='float32')

    out2b_channel_pad = input_3b_ndim[3] - output_1b_ndim[3]
    out2b = np.zeros(input_3b_ndim, dtype='float32')
    out4b_channel_pad = input_5b_ndim[3] - output_3b_ndim[3]
    out4b = np.zeros(input_5b_ndim, dtype='float32')
    out6b_channel_pad = input_7b_ndim[3] - output_5b_ndim[3]
    out6b = np.zeros(input_7b_ndim, dtype='float32')
    #if DEBUG:
    if False:
        print("output_1b_ndim: ", output_1b_ndim)
        print("output_2b_channel_pad: ",  out2b_channel_pad)
        print("output_2b_ndim: ", input_3b_ndim)
        print("output_3b_ndim: ", output_3b_ndim)
        print("output_4b_channel_pad: ",  out4b_channel_pad)
        print("output_4b_ndim: ", input_5b_ndim)
        print("output_5b_ndim: ", output_5b_ndim)
        print("output_6b_channel_pad: ",  out6b_channel_pad)
        print("output_6b_ndim: ", input_7b_ndim)

    #if DEBUG :
    if False:
        print(" inputTensor1={}\n".format( inputTensor_1[0]))
        print("outputTensor1[0]={}\n".format(outputTensor_1[0]))
        print("outputTensor1[1]={}\n".format(outputTensor_1[1]))
        print(" inputTensor3[0]={}\n".format( inputTensor_3[0]))
        print(" inputTensor3[1]={}\n".format( inputTensor_3[1]))
        print("outputTensor3[0]={}\n".format(outputTensor_3[0]))
        print("outputTensor3[1]={}\n".format(outputTensor_3[1]))
        print(" inputTensor5[0]={}\n".format( inputTensor_5[0]))
        print(" inputTensor5[1]={}\n".format( inputTensor_5[1]))
        print("outputTensor5[0]={}\n".format(outputTensor_5[0]))
        print("outputTensor5[1]={}\n".format(outputTensor_5[1]))
        print(" inputTensor7[0]={}\n".format( inputTensor_7[0]))
        print(" inputTensor7[1]={}\n".format( inputTensor_7[1]))
        print("outputTensor7[0]={}\n".format(outputTensor_7[0]))
        print("outputTensor7[1]={}\n".format(outputTensor_7[1]))

    #if not imgQ.empty():
    #    img_org = imgQ.get()
    if True:
        
        # run DPU
# inputTensor1
# [<xir.Tensor named 'BlazePalm__input_0_fix'>]
#outputTensor1
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__MaxPool2d_max_pool__13016_fix'>]
        if DEBUG:
            print("Executing Task 1 [DPU] ...")
        execute_async(
            dpu_1, {
          "BlazePalm__input_0_fix": inputData[0],
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix": out1a,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__MaxPool2d_max_pool__13016_fix": out1b
            })
        if DEBUG:
            print("out1a.shape = ",out1a.shape)
            print("out1b.shape = ",out1b.shape)
        if DEBUG:
            print("Executing Task 2 [CPU] ...")
        inp2b = out1b.copy()
        #out2b = Tanh(inp2b)
        out2b.fill(0.)
        out2b[:,:,:,0:out2b_channel_pad] = inp2b
        if DEBUG:
            print("inp2b.shape = ",inp2b.shape)
            print("out2b.shape = ",out2b.shape)
        # run DPU
# inputTensor3
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__ret_51_fix'>]
#outputTensor3
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__MaxPool2d_max_pool__13404_fix'>]
        if DEBUG:
            print("Executing Task 3 [DPU] ...")
        execute_async(
            dpu_3, {
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_8__ReLU_act__12992_fix": out1a,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__ret_51_fix": out2b,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix": out3a,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__MaxPool2d_max_pool__13404_fix": out3b
            })
        if DEBUG:
            print("out3a.shape = ",out3a.shape)
            print("out3b.shape = ",out3b.shape)
        if DEBUG:
            print("Executing Task 4 [CPU] ...")
        inp4b = out3b.copy()
        #out4b = Sigmoid1(inp4b)
        out4b.fill(0.)
        out4b[:,:,:,0:out4b_channel_pad] = inp4b
        if DEBUG:
            print("inp4b.shape = ",inp4b.shape)
            print("out4b.shape = ",out4b.shape)
        # run DPU
# inputTensor5
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__ret_103_fix'>]
#outputTensor5
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__MaxPool2d_max_pool__13792_fix'>]
        if DEBUG:
            print("out5a.shape = ",out5a.shape)
            print("out5b.shape = ",out5b.shape)
        if DEBUG:
            print("Executing Task 5 [DPU] ...")
        execute_async(
            dpu_5, {
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_16__ReLU_act__13380_fix": out3a,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__ret_103_fix": out4b,
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix":   out5a,
          "BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__MaxPool2d_max_pool__13792_fix":   out5b
            })
        if DEBUG:
            print("Executing Task 6 [CPU] ...")
        inp6b = out5b.copy()
        out6b.fill(0.)
        out6b[:,:,:,0:out6b_channel_pad] = inp6b
        if DEBUG:
            print("inp6b.shape = ",inp6b.shape)
            print("out6b.shape = ",out6b.shape)
        
# inputTensor7
# [<xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__ret_155_fix'>]
#outputTensor7
# [<xir.Tensor named 'BlazePalm__BlazePalm_ret_293_fix'>, <xir.Tensor named 'BlazePalm__BlazePalm_ret_fix'>]
        if DEBUG:
            print("out7a.shape = ",out7a.shape)
            print("out7b.shape = ",out7b.shape)
        if DEBUG:
            print("Executing Task 7 [DPU] ...")
        execute_async(
            dpu_7, {
          "BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_24__ReLU_act__13768_fix": out5a,
          "BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__ret_155_fix": out6b,
          "BlazePalm__BlazePalm_ret_293_fix":   out7a,
          "BlazePalm__BlazePalm_ret_fix":   out7b
            })
        if DEBUG:
            print("Executing Task 8 [CPU] ...")
        palm_out1 = out7a.copy()
        palm_out2 = out7b.copy()
        if DEBUG:
            print("palm_out1.shape = ",palm_out1.shape)
            print("palm_out2.shape = ",palm_out2.shape)
        if DEBUG:
            print("Execution Done ...")
        
        return palm_out1, palm_out2

#def detection2roi(self, detection):
detection2roi_method = 'box'
kp1 = 0
kp2 = 2
theta0 = np.pi/2
dscale = 2.6
dy = -0.5
def detection2roi(detection):
    """ Convert detections from detector to an oriented bounding box.

    Adapted from:
    # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

    The center and size of the box is calculated from the center 
    of the detected box. Rotation is calcualted from the vector
    between kp1 and kp2 relative to theta0. The box is scaled
    and shifted by dscale and dy.

    """
    if detection2roi_method == 'box':
        # compute box center and scale
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        xc = (detection[:,1] + detection[:,3]) / 2
        yc = (detection[:,0] + detection[:,2]) / 2
        scale = (detection[:,3] - detection[:,1]) # assumes square boxes

    elif detection2roi_method == 'alignment':
        # compute box center and scale
        # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        xc = detection[:,4+2*kp1]
        yc = detection[:,4+2*kp1+1]
        x1 = detection[:,4+2*kp2]
        y1 = detection[:,4+2*kp2+1]
        scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
    else:
        raise NotImplementedError(
            "detection2roi_method [%s] not supported"%detection2roi_method)

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:,4+2*kp1]
    y0 = detection[:,4+2*kp1+1]
    x1 = detection[:,4+2*kp2]
    y1 = detection[:,4+2*kp2+1]
    #theta = torch.atan2(y0-y1, x0-x1) - theta0
    theta = np.arctan2(y0-y1, x0-x1) - theta0
    #print("[detection2roi] xc.shape=", xc.shape)
    #print("[detection2roi] yc.shape=", yc.shape)
    #print("[detection2roi] theta.shape=", theta.shape)
    #print("[detection2roi] scale.shape=", scale.shape)
    return xc, yc, scale, theta


#def extract_roi(self, frame, xc, yc, theta, scale):
resolution = 256
def extract_roi(frame, xc, yc, theta, scale):
    #print("[extract_roi] frame.shape = ", frame.shape," frame.dtype=", frame.dtype)
    #print("[extract_roi] xc.shape=", xc.shape," xc.dtype=", xc.dtype," xc=",xc)
    #print("[extract_roi] yc.shape=", yc.shape," yc.dtype=", yc.dtype," yc=",yc)
    #print("[extract_roi] theta.shape=", theta.shape," theta.dtype=", theta.dtype," theta=",theta)
    #print("[extract_roi] scale.shape=", scale.shape," scale.dtype=", scale.dtype," scale=",scale)

    # Assuming scale is a NumPy array of size [N]
    scaleN = scale.reshape(-1, 1, 1).astype(np.float32)
    #print("[extract_roi] scaleN.shape=", scaleN.shape," scaleN.dtype=", scaleN.dtype)
    #print("[extract_roi] scaleN=", scaleN)

    # Define points
    points = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=np.float32)
    #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
    #print("[extract_roi] points=", points)

    # Element-wise multiplication
    points = points * scaleN / 2
    points = points.astype(np.float32)
    #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
    #print("[extract_roi] points=", points)

    #R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R = np.zeros((theta.shape[0],2,2),dtype=np.float32)
    for i in range (theta.shape[0]):
        R[i,:,:] = [[np.cos(theta[i]), -np.sin(theta[i])], [np.sin(theta[i]), np.cos(theta[i])]]
    #print("[extract_roi] R.shape=", R.shape," R=",R)

    center = np.column_stack((xc, yc))
    #print("[extract_roi] center.shape=", center.shape," center=",center)
    center = np.expand_dims(center, axis=-1)
    #print("[extract_roi] center.shape=", center.shape," center=",center)

    #points = np.matmul(R, points) + center[:, None]
    points = np.matmul(R, points) + center
    points = points.astype(np.float32)
    #print("[extract_roi] points.shape=", points.shape," point.dtype=",points.dtype)
    #print("[extract_roi] points=", points)

    res = resolution
    points1 = np.array([[0, 0], [0, res-1], [res-1, 0]], dtype=np.float32)
    #print("[extract_roi] points1.shape=", points1.shape)
    #print("[extract_roi] points1=", points1)

    affines = []
    imgs = []

    for i in range(points.shape[0]):
        pts = points[i,:,:3].T
        #print("[extract_roi] points.shape=", points.shape," points=", points)
        #print("[extract_roi] pts.shape=", pts.shape," pts=", pts)
        #print("[extract_roi] points1.shape=", points1.shape," points1=", points1)
        #print("[extract_roi] pts.dtype=",pts.dtype)
        #print("[extract_roi] points1.dtype=",points1.dtype)
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res, res))  # No borderValue in NumPy
        img = img.astype('float32') / 255.0
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)

    if imgs:
        imgs = np.stack(imgs).transpose(0, 3, 1, 2).astype('float32')
        affines = np.stack(affines).astype('float32')
    else:
        imgs = np.zeros((0, 3, res, res), dtype='float32')
        affines = np.zeros((0, 2, 3), dtype='float32')

    return imgs, affines, points


#def _decode_boxes(self, raw_boxes, anchors):
x_scale = 256
y_scale = 256
h_scale = 256
w_scale = 256
num_keypoints = 7
def _decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    if DEBUG:
        print("[DEBUG] _decode_boxes ... entering")
         
    #boxes = torch.zeros_like(raw_boxes)
    boxes = np.zeros( raw_boxes.shape )

    x_center = raw_boxes[..., 0] / x_scale * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(num_keypoints):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    if DEBUG:
        print("[DEBUG] _decode_boxes ... done")

    return boxes

#def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
score_clipping_thresh = 100.0
#min_score_thresh = 0.5 # too many false positives
#min_score_thresh = 0.8 # still has false positives
min_score_thresh = 0.7
def _tensors_to_detections( raw_box_tensor, raw_score_tensor, anchors):
    if DEBUG:
        print("[DEBUG] _tensors_to_detections ... entering")

    detection_boxes = _decode_boxes(raw_box_tensor, anchors)
    #detection_boxes = detection_boxes[0]
	    
    if DEBUG:
       print("[DEBUG] _tensors_to_detections ... thresholding")

    thresh = score_clipping_thresh
    #raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
    raw_score_tensor = np.clip(raw_score_tensor,-thresh,thresh)
    #detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
    detection_scores = 1/(1 + np.exp(-raw_score_tensor))
    detection_scores = np.squeeze(detection_scores, axis=-1)        

    if DEBUG:
       print("[DEBUG] _tensors_to_detections ... min score thresholding (",min_score_thresh,")")
	    
    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    #mask = detection_scores >= 0.7
    mask = detection_scores >= min_score_thresh

    #print(raw_box_tensor.shape, raw_score_tensor.shape)
    #(14, 2944, 18) (14, 2944, 1)
    #print(detection_boxes.shape, detection_scores.shape, mask.shape)
    #(14, 2944, 18) (14, 2944) (14, 2944)         


    if DEBUG:
       print("[DEBUG] _tensors_to_detections ... processing loop")

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []
    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        #scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        scores = detection_scores[i, mask[i]]
        scores = np.expand_dims(scores,axis=-1)         
        #print(raw_box_tensor.shape,boxes.shape, raw_score_tensor.shape,scores.shape)         
        #(14, 2944, 18) (0, 18) (14, 2944, 1) (0, 1)       
        #output_detections.append(torch.cat((boxes, scores), dim=-1))
        boxes_scores = np.concatenate((boxes,scores),axis=-1)
        #print(boxes_scores.shape)         
        #(0, 19)       
        output_detections.append(boxes_scores)

    if DEBUG:
       print("[DEBUG] _tensors_to_detections ... done")

    return output_detections

# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    #A = box_a.size(0)
    #B = box_b.size(0)
    #max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
    #                   box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
    #                   box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    #inter = torch.clamp((max_xy - min_xy), min=0)
    #return inter[:, :, 0] * inter[:, :, 1]

    # This NumPy version follows a similar approach to the PyTorch code, 
    # using broadcasting and element-wise operations to compute the intersection area. 
    # Note that in NumPy, you use np.minimum, np.maximum, and np.clip 
    # instead of torch.min, torch.max, and torch.clamp. 
    # Also, the unsqueeze operation is replaced with np.expand_dims, 
    # and the expand operation is achieved using repeat.
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_xy = np.minimum(
        np.expand_dims(box_a[:, 2:], axis=1).repeat(B, axis=1),
        np.expand_dims(box_b[:, 2:], axis=0).repeat(A, axis=0)
    )

    min_xy = np.maximum(
        np.expand_dims(box_a[:, :2], axis=1).repeat(B, axis=1),
        np.expand_dims(box_b[:, :2], axis=0).repeat(A, axis=0)
    )

    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

    return inter[:, :, 0] * inter[:, :, 1]    

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A n B / A ? B = A n B / (area(A) + area(B) - A n B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    #inter = intersect(box_a, box_b)
    #area_a = ((box_a[:, 2]-box_a[:, 0]) *
    #          (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    #area_b = ((box_b[:, 2]-box_b[:, 0]) *
    #          (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    #union = area_a + area_b - inter
    #return inter / union  # [A,B]

    # In this NumPy version, unsqueeze is replaced with reshape, 
    # and expand_as is replaced with repeat to handle the expansion along the appropriate dimensions. 
    # The general structure and logic of the code remain the same.
    inter = intersect(box_a, box_b)

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).reshape(-1, 1).repeat(box_b.shape[0], axis=1)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).reshape(1, -1).repeat(box_a.shape[0], axis=0)

    union = area_a + area_b - inter

    return inter / union    

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    #return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)
    
    # In this NumPy version, unsqueeze is replaced with np.expand_dims, 
    # and squeeze is used to remove the singleton dimension after calculating the Jaccard overlap 
    # using the previously defined jaccard function. The structure of the function remains the same.
    return jaccard(np.expand_dims(box, axis=0), other_boxes).squeeze(0)

def _weighted_non_max_suppression_(detections):
	    """The alternative NMS method as mentioned in the BlazeFace paper:
	    "We replace the suppression algorithm with a blending strategy that
	    estimates the regression parameters of a bounding box as a weighted
	    mean between the overlapping predictions."
	    The original MediaPipe code assigns the score of the most confident
	    detection to the weighted detection, but we take the average score
	    of the overlapping detections.
	    The input detections should be a Tensor of shape (count, 17).
	    Returns a list of PyTorch tensors, one for each detected face.
	    
	    This is based on the source code from:
	    mediapipe/calculators/util/non_max_suppression_calculator.cc
	    mediapipe/calculators/util/non_max_suppression_calculator.proto
	    """
	    if DEBUG:
	       print("[DEBUG] _weighted_non_max_suppression ... entering")
	    if len(detections) == 0: return []

	    output_detections = []

	    # Sort the detections from highest to lowest score.
	    #remaining = torch.argsort(detections[:, 18], descending=True)
	    remaining = np.argsort(detections[:, 18])[::-1]

	    while len(remaining) > 0:
	        detection = detections[remaining[0]]

	        # Compute the overlap between the first box and the other 
	        # remaining boxes. (Note that the other_boxes also include
	        # the first_box.)
	        first_box = detection[:4]
	        other_boxes = detections[remaining, :4]
	        ious = overlap_similarity(first_box, other_boxes)

	        # If two detections don't overlap enough, they are considered
	        # to be from different faces.
	        mask = ious >= 0.3
	        overlapping = remaining[mask]
	        remaining = remaining[~mask]

	        # Take an average of the coordinates from the overlapping
	        # detections, weighted by their confidence scores.
	        #weighted_detection = detection.clone()
	        weighted_detection = np.copy(detection)
	        if len(overlapping) > 1:
	            coordinates = detections[overlapping, :18]
	            scores = detections[overlapping, 18:19]
	            total_score = scores.sum()
	            #weighted = (coordinates * scores).sum(dim=0) / total_score
	            weighted = np.sum(coordinates * scores, axis=0) / total_score                       
	            weighted_detection[:18] = weighted
	            weighted_detection[18] = total_score / len(overlapping)

	        output_detections.append(weighted_detection)

	    if DEBUG:
	       print("[DEBUG] _weighted_non_max_suppression ... done")

	    return output_detections


#def _weighted_non_max_suppression(self, detections):
num_coords = 18
min_suppression_threshold = 0.3
def _weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Tensor of shape (count, 17).

    Returns a list of PyTorch tensors, one for each detected face.
        
    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    """
    if len(detections) == 0: return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    #remaining = torch.argsort(detections[:, num_coords], descending=True)
    remaining = np.argsort(detections[:, num_coords])[::-1]    

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other 
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > min_suppression_threshold
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        #weighted_detection = detection.clone()
        weighted_detection = np.copy(detection)
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :num_coords]
            scores = detections[overlapping, num_coords:num_coords+1]
            total_score = scores.sum()
            #weighted = (coordinates * scores).sum(dim=0) / total_score
            weighted = np.sum(coordinates * scores, axis=0) / total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections    



def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0]
        padh = 0
        padw = 256 - w1
        scale = size0[1] / w1
    else:
        h1 = 256 * size0[0] // size0[1]
        w1 = 256
        padh = 256 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad


def denormalize_detections(detections, scale, pad):
    """ maps detection coordinates from [0,1] to image coordinates

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio. This function maps the
    normalized coordinates back to the original image coordinates.

    Inputs:
        detections: nxm tensor. n is the number of detections.
            m is 4+2*k where the first 4 valuse are the bounding
            box coordinates and k is the number of additional
            keypoints output by the detector.
        scale: scalar that was used to resize the image
        pad: padding in the x and y dimensions

    """
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections

def draw_detections(img, detections, with_keypoints=True):
    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1) 

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)


def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


anchors = np.load('./anchors_palm.npy')
print('[INFO] Loading anchors of size ',anchors.shape)

# Get a list of subgraphs from the compiled model file
g = xir.Graph.deserialize(model_path)
#print(g)
#{name: 'BlazePalm', 
# op_num: 349, 
# attrs: {
#    'files_md5sum': {
#       '/workspace/blazepalm_tutorial/vitis_ai/./quantize_result/BlazePalm_int.xmodel': '745c75900177a575fd1549f51e6a09b2'}, 
#       'libs_info': {
#          'xcompiler.3.5.0': 'be7bc16b739398070b4526571b1d251757bf0e57', 
#          'xcompiler.3.5.0 : target-factory.3.5.0': '947d287c09dadab682bea1c60ae5edf21fbcbe64', 
#          'xcompiler.3.5.0 : xir.3.5.0': 'ea490eebe8414766bebf74622ca6bff0ea57b6d8'
#       }
#    }
# }

subgraphs = g.get_root_subgraph().toposort_child_subgraph()
#print(subgraphs)
#[
#  <xir.Subgraph named 'subgraph_BlazePalm__input_0'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_2__Sequential_convs__Conv2d_0__ret_7'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_9__ret_51'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_10__Sequential_convs__Conv2d_0__ret_59'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__ret_103'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__Sequential_convs__Conv2d_0__ret_105'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_Sequential_backbone2__BlazeBlock_0__ret_155'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_BlazeBlock_blaze1__Sequential_convs__Conv2d_0__ret_259'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_ret_fix_'>, 
#  <xir.Subgraph named 'subgraph_BlazePalm__BlazePalm_ret_293_fix_'>
#]
dpu_subgraph0 = subgraphs[0]
dpu_subgraph1 = subgraphs[1]
dpu_subgraph3 = subgraphs[3]
dpu_subgraph5 = subgraphs[5]
dpu_subgraph7 = subgraphs[7]

if DEBUG:
    print("dpu_subgraph0 = " + dpu_subgraph0.get_name()) #dpu_subgraph0 = subgraph_BlazePalm__input_0
    print("dpu_subgraph1 = " + dpu_subgraph1.get_name()) #dpu_subgraph1 = subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_2__Sequential_convs__Conv2d_0__ret_7
    print("dpu_subgraph3 = " + dpu_subgraph3.get_name()) #dpu_subgraph3 = subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_10__Sequential_convs__Conv2d_0__ret_59
    print("dpu_subgraph5 = " + dpu_subgraph5.get_name()) #dpu_subgraph5 = subgraph_BlazePalm__BlazePalm_Sequential_backbone1__BlazeBlock_17__Sequential_convs__Conv2d_0__ret_105
    print("dpu_subgraph7 = " + dpu_subgraph7.get_name()) #dpu_subgraph7 = subgraph_BlazePalm__BlazePalm_BlazeBlock_blaze1__Sequential_convs__Conv2d_0__ret_259

dpu_1 = vart.Runner.create_runner(dpu_subgraph1, "run")
dpu_3 = vart.Runner.create_runner(dpu_subgraph3, "run")
dpu_5 = vart.Runner.create_runner(dpu_subgraph5, "run")
dpu_7 = vart.Runner.create_runner(dpu_subgraph7, "run")

inputTensor_1  = dpu_1.get_input_tensors()
outputTensor_7 = dpu_7.get_output_tensors()

# Get input scaling
input_fixpos = inputTensor_1[0].get_attr("fix_point")
input_scale = 2**input_fixpos
print('[INFO] input_fixpos=',input_fixpos,' input_scale=',input_scale)

# Get input/output tensors dimensions
inputShape = tuple(inputTensor_1[0].dims)
outputShape1 = tuple(outputTensor_7[0].dims)
outputShape2 = tuple(outputTensor_7[1].dims)
batchSize = inputShape[0]
    
#batchSize = 1
#inputShape   = (batchSize,) + inputShape[1:]
#outputShape1 = (batchSize,) + outputShape1[1:]
#outputShape2 = (batchSize,) + outputShape2[1:]

print('[INFO] batch size = ',batchSize)
print('[INFO] input dimensions = ',inputShape)
print('[INFO] output dimensions = ',outputShape1,outputShape2)

print("================================")
print("Palm Detection Demo:")
print("\tPress ESC to quit ...")
print("\tPress 't' to toggle between image and live video")
print("\tPress 'p' to pause video ...")
print("\tPress 'c' to continue ...")
print("\tPress 's' to step one frame at a time ...")
print("\tPress 'w' to take a photo ...")
print("================================")

bStep = False
bPause = False
bUseImage = False

image = []
output = []

frame_count = 0

# init the real-time FPS counter
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)

while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    #if cap.grab():
    if True:
        frame_count = frame_count + 1
        #flag, image = cap.retrieve()
        flag, image = cap.read()
        if not flag:
            break
        else:
            if bUseImage:
                image = cv2.imread('./image.jpg')
                
            #image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            
            # BlazePalm pre-processing
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            model_input,img2,scale,pad=resize_pad(image)
            
            # mediapipe blazepalm model expects data in D,3,W,H format
            #model_input = np.transpose(np.array(model_input),(0,3,2,1))            
            
            # x.float() / 127.5 - 1.0
            model_input = (model_input / 127.5) - 1.0
                        
            #try:
            if True:
                # Reformat from model_input of size (256,256,3) to model_x of size (1,256,256,3)
                model_x = []
                model_x.append( model_input )
                model_x = np.array(model_x)
            
                """ Prepare input/output buffers """
                if DEBUG:
                    print("[INFO] PalmDetector - prep input buffer ")
                inputData = []
                inputData.append(np.empty((inputShape),dtype=np.float32,order='C'))
                inputImage = inputData[0]
                inputImage[0,...] = model_x
                               
                if DEBUG:
                    print("[INFO] PalmDetector - prep output buffer ")
                outputData = []
                outputData.append(np.empty((outputShape1),dtype=np.float32,order='C'))
                outputData.append(np.empty((outputShape2),dtype=np.float32,order='C'))

                """ Execute model on DPU """
                if DEBUG:
                    print("[INFO] PalmDetector - execute ")
                palm_out1, palm_out2 = DEBUG_runDPU(dpu_1, dpu_3, dpu_5, dpu_7, inputData, outputData )
                #if DEBUG:
                #    print(palm_out1,palm_out2)
                
                # 3. Postprocess the raw predictions:
                if DEBUG:
                    print("[INFO] PalmDetector - post-processing - extracting detections ")
                detections = _tensors_to_detections(palm_out2, palm_out1, anchors)

                # 4. Non-maximum suppression to remove overlapping detections:
                if DEBUG:
                    print("[INFO] PalmDetector - post-processing - non-maxima suppression ")
                filtered_detections = []
                for i in range(len(detections)):
                    palms = _weighted_non_max_suppression(detections[i])
                    if len(palms) > 0:
                      filtered_detections.append(palms)

                if DEBUG:
                    print("[INFO] PalmDetector - post-processing - filtered_detections = ",len(filtered_detections),filtered_detections)
                
                if len(filtered_detections) > 0:
                    normalized_detections = np.array(filtered_detections)[0]
                    if DEBUG:
                        print("[INFO] PalmDetector - post-processing - normalized_detections = ",normalized_detections.shape,normalized_detections)

                
                    palm_detections = denormalize_detections(normalized_detections,scale,pad)
                    
                    xc,yc,scale,theta = detection2roi(palm_detections)
                    hand_img,hand_affine,hand_box = extract_roi(image,xc,yc,theta,scale)
                    
                    draw_roi(output,hand_box)
                    draw_detections(output,palm_detections)
                    

                # BlazePalm post-processing
                if DEBUG:
                    print("[INFO] PalmDetector - post-processing ")

                if DEBUG:
                    print("[INFO] PalmDetector - done ")
                        
            #except:
            #    print("ERROR : Exception occured during Palm detection ...")

                         
                
            # display real-time FPS counter (if valid)
            if rt_fps_valid == True:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
            
            # show the output image
            cv2.imshow("Palm Detection", output)

    if bStep == True:
        key = cv2.waitKey(0)
    elif bPause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(10)

    #print(key)
    
    if key == 119: # 'w'
        filename = ("frame%04d_palm%02d.tif"%(frame_count,asl_id))
            
        print("Capturing ",filename," ...")
        cv2.imwrite(os.path.join(output_dir,filename),roi_img)
       
    if key == 115: # 's'
        bStep = True    
    
    if key == 112: # 'p'
        bPause = not bPause

    if key == 99: # 'c'
        bStep = False
        bPause = False
        
    if key == 116: # 't'
        bUseImage = not bUseImage  

    if key == 27 or key == 113: # ESC or 'q':
        break

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = 1
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #print("[INFO] ",rt_fps_message)
        rt_fps_count = 0



# Stop the Palm Detection
del dpu_1
del dpu_3
del dpu_5
del dpu_7

# Cleanup
cv2.destroyAllWindows()
