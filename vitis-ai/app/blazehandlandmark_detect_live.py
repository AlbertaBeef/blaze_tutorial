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
#   https://www.github.com/AlbertaBeef/blaze_tutorial/tree/2023.1
#   https://github.com/Xilinx/Vitis-AI/blob/master/examples/custom_operator/pytorch_example/deployment/python/pointpillars_main.py
#
# Dependencies:
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

from ctypes import *
from typing import List
#import xir
import pathlib
#import vitis_ai_library
#import threading
import time
import sys
import argparse
import glob
import subprocess
import re

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS


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

cv2.namedWindow('BlazeHandLandmark demo')


# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("camera",input_video," (",frame_width,",",frame_height,")")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()  
ap.add_argument('-m', '--model1',     type=str, default='blazepalm.xmodel', help='Path of blazepalm xmodel. Default is blazepalm.xmodel')
ap.add_argument('-n', '--model2',     type=str, default='blazepalm.xmodel', help='Path of blazehandlardmark.xmodel. Default is blazehandlardmark.xmodel')

args = ap.parse_args()  
  
print ('Command line options:')
print (' --model1     : ', args.model1)
print (' --model2     : ', args.model2)

#dpu_arch = detect_dpu_architecture()
#print('[INFO] Detected DPU architecture : ',dpu_arch)
#
#model_path = './model_1/'+dpu_arch+'/asl_classifier_1.xmodel'
#print('[INFO] ASL model : ',model_path)


palm_detector = BlazePalm()
palm_detector.load_xmodel(args.model1,debug=False)
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark()
hand_regressor.load_xmodel(args.model2,debug=False)


print("================================================================")
print("Palm Detection Demo:")
print("================================================================")
print("\tPress ESC to quit ...")
print("----------------------------------------------------------------")
print("\tPress 'p' to pause video ...")
print("\tPress 'c' to continue ...")
print("\tPress 's' to step one frame at a time ...")
print("\tPress 'w' to take a photo ...")
print("----------------------------------------------------------------")
print("\tPress 't' to toggle between image and live video")
print("\tPress 'd' to toggle debug image on/off")
print("\tPress 'v' to toggle verbose on/off")
print("================================================================")

bStep = False
bPause = False
bUseImage = False
bShowDebugImage = False
bVerbose = False

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
            
            # BlazePalm pipeline
            
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img1,img2,scale,pad=resize_pad(image)
            #print("[INFO] img1.shape=",img1.shape, " img1.dtype=",img1.dtype)
            #print("[INFO] img2.shape=",img2.shape, " img2.dtype=",img2.dtype)
            #print("[INFO] scale=",scale)
            #print("[INFO] pad=",pad)
            
            normalized_detections = palm_detector.predict_on_image(img1)
            if len(normalized_detections) > 0:
            
                palm_detections = denormalize_detections(normalized_detections,scale,pad)
                    
                xc,yc,scale,theta = palm_detector.detection2roi(palm_detections)
                #print("[INFO] xc.shape=",xc.shape, " xc=",xc)
                #print("[INFO] yc.shape=",yc.shape, " yc=",yc)
                #print("[INFO] scale.shape=",scale.shape, " scale=",scale)
                #print("[INFO] theta.shape=",theta.shape, " theta=",theta)
                hand_img,hand_affine,hand_box = hand_regressor.extract_roi(image,xc,yc,theta,scale)
                #print("[INFO] hand_img.shape=",hand_img.shape, " hand_img.dtype=",hand_img.dtype)
                #print("[INFO] hand_affine.shape=",hand_affine.shape, " hand_affine=",hand_affine)
                #print("[INFO] hand_box.shape=",hand_box.shape, " hand_box=",hand_box)

                if bShowDebugImage:
                    # show the output image
                    debug_img = img1.astype(np.float32)/255.0
                    for i in range(hand_img.shape[0]):
                        debug_img = cv2.hconcat([debug_img,hand_img[i]])
                    debug_img = cv2.cvtColor(debug_img,cv2.COLOR_RGB2BGR)
                    cv2.imshow("Debug", debug_img)
                
                flags, handed, normalized_landmarks = hand_regressor.predict_landmarks(hand_img)
                #print("[INFO] flags.shape=",flags.shape, " flags=",flags)
                #print("[INFO] handed.shape=",handed.shape, " handed=",handed)
                #print("[INFO] normalized_landmarks.shape=",normalized_landmarks.shape, " normalized_landmarks=",normalized_landmarks)
                landmarks = hand_regressor.denormalize_landmarks(normalized_landmarks, hand_affine)
                #print("[INFO] landmarks.shape=",landmarks.shape, " landmarks=",landmarks)

                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if True: #flag>.5:
                        draw_landmarks(output, landmark[:,:2], HAND_CONNECTIONS, size=2)                
                   
                draw_roi(output,hand_box)
                draw_detections(output,palm_detections)

                        
                
            # display real-time FPS counter (if valid)
            if rt_fps_valid == True:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
            
            # show the output image
            cv2.imshow("BlazeHandLandmark demo", output)

    if bStep == True:
        key = cv2.waitKey(0)
    elif bPause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(10)

    #print(key)
    
    if key == 119: # 'w'
        filename = ("blazehandlandmark_frame%04d_input.tif"%(frame_count))
        print("Capturing ",filename," ...")
        input_img = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir,filename),input_img)

        filename = ("blazehandlandmark_frame%04d_detection.tif"%(frame_count))
        print("Capturing ",filename," ...")
        cv2.imwrite(os.path.join(output_dir,filename),output)
        
        if bShowDebugImage:
            filename = ("blazehandlandmark_frame%04d_debug.tif"%(frame_count))
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),debug_img)
       
    if key == 115: # 's'
        bStep = True    
    
    if key == 112: # 'p'
        bPause = not bPause

    if key == 99: # 'c'
        bStep = False
        bPause = False
        
    if key == 116: # 't'
        bUseImage = not bUseImage  

    if key == 100: # 'd'
        bShowDebugImage = not bShowDebugImage  

    if key == 27 or key == 113: # ESC or 'q':
        break

    if key == 118: # 'v'
        bVerbose = not bVerbose 
        palm_detector.DEBUG = bVerbose 
        hand_regressor.DEBUG = bVerbose

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

# Cleanup
cv2.destroyAllWindows()
