import numpy as np
import cv2
import argparse
import os

from memryx import NeuralCompiler

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch'       , type=str,  default="mx3", help="Hailo HW architecture.  Default is 'mx3'.")
ap.add_argument('-b', '--blaze'      , type=str,  default="hand", help="Blaze application ('hand', 'face', 'pose').  Default is 'hand'")
ap.add_argument('-n', '--name'       , type=str,  default="palm_detection_lite", help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-m', '--model'      , type=str,  default="models/palm_detection_lite.tflite", help="Detection Model (TF-Lite). Default is 'models/palm_detection_lite.tflite'")
ap.add_argument('-r', '--resolution' , type=int,  default=192, help="Input resolution.  Default is 192 for 192x192.")
ap.add_argument('-p', '--process'    , type=str,  default="all", help="Command seperated list of processes to run ( 'inspect', 'compile', 'all'=='compile' ). Default is 'all'")


args = ap.parse_args()  
  
print('Command line options:')
print(' --arch        : ', args.arch)
print(' --blaze       : ', args.blaze)
print(' --name        : ', args.name)
print(' --model       : ', args.model)
print(' --resolution  : ', args.resolution)
print(' --process     : ', args.process)


#
# Cropping
# Reference : https://developer.memryx.com/tutorials/how_to/cropping.html
#


if ("inspect" in args.process):

    arch_type = args.arch
    num_chips = 0
    if arch_type == "mx3":
        num_chips = 4
        print("[INFO] Targetting arch : ",arch_type," (",num_chips,")")    
    else:
        print("[ERROR] Unknown arch : ",arch_type)    

    model_path = args.model
    model_name = args.name
    
    file_name, model_type = os.path.splitext(model_path)
    if model_type == ".tflite":
        print("[INFO] TensorFlow-lite model : ",model_path)    
    elif model_type == ".onnx":
        print("[ERROR] ONNX model : ",model_path)    
    else:
        print("[ERROR] Unknown model type : ",model_type)    
        
    #nc = NeuralCompiler(num_chips=4, models=model_path, verbose=1, dfp_fname=None, effort="Lazy", show_optimization=False, autocrop=True, model_in_out=model_name+'.json')
    nc = NeuralCompiler(num_chips=4, models=model_path, verbose=1, dfp_fname=None, effort="Lazy", show_optimization=False, autocrop=True)
    dfp = nc.run()

#
# Compile
# Reference : https://developer.memryx.com/tutorials/how_to/compile_benchmark_api.html
#


if ("compile" in args.process or args.process == "all"):

    arch_type = args.arch
    num_chips = 0
    if arch_type == "mx3":
        num_chips = 4
        print("[INFO] Targetting arch : ",arch_type," (",num_chips,")")    
    else:
        print("[ERROR] Unknown arch : ",arch_type)    

    model_path = args.model
    model_name = args.name
    model_size = args.resolution

    file_name, model_type = os.path.splitext(model_path)
    if model_type == ".tflite":
        print("[INFO] TensorFlow-lite model : ",model_path) 
            
        if model_name == "palm_detection_v0_07":
            assert (model_size==256), "palm_detection_v0_07 resolution should be 256"
            #start_node_names = ['input_1']
            #end_node_names = ['classificator_8/BiasAdd', 'classificator_16/BiasAdd', 'classificator_32/BiasAdd', 'regressor_8/BiasAdd', 'regressor_16/BiasAdd', 'regressor_32/BiasAdd']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [256, 256, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'classificator_8/BiasAdd', 'shape': [32, 32, 1, 2]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'classificator_16/BiasAdd', 'shape': [16, 16, 1, 2]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'classificator_32/BiasAdd', 'shape': [8, 8, 1, 6]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'regressor_8/BiasAdd', 'shape': [32, 32, 1, 36]}
            #MPU 3 output port 4: {'model_index': 0, 'layer_name': 'regressor_16/BiasAdd', 'shape': [16, 16, 1, 36]}
            #MPU 3 output port 5: {'model_index': 0, 'layer_name': 'regressor_32/BiasAdd', 'shape': [8, 8, 1, 108]}
            inputs = "input_1"
            outputs = "classificator_8/BiasAdd,classificator_16/BiasAdd,classificator_32/BiasAdd,regressor_8/BiasAdd,regressor_16/BiasAdd,regressor_32/BiasAdd"
        elif model_name == "hand_landmark_v0_07":
            assert (model_size==256), "hand_landmark_v0_07 resolution should be 256"
            #start_node_names = ['input_1']
            #end_node_names = ['convld_21_3d','activation_handflag','activation_handedness']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [256, 256, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'conv_handflag', 'shape': [1, 1, 1, 1]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'conv_handedness', 'shape': [1, 1, 1, 1]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'convld_21_3d', 'shape': [1, 1, 1, 63]}
            inputs = "input_1"
            outputs = "convld_21_3d,conv_handflag,conv_handedness"
        elif model_name == "palm_detection_lite" or model_name == "palm_detection_full":
            assert (model_size==192), "palm_detection_lite/full resolution should be 192"
            #start_node_names = ['input_1']
            #end_node_names = [
            #    'model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [192, 192, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [12, 12, 1, 6]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [12, 12, 1, 108]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [24, 24, 1, 2]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [24, 24, 1, 36]}
            inputs = "input_1"
            outputs = "model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1"
        elif model_name == "hand_landmark_lite" or model_name == "hand_landmark_full":
            assert (model_size==224), "hand_landmark_lite/full resolution should be 192"
            #start_node_names = ['input_1']
            #end_node_names = ['Identity','Identity_1','Identity_2','Identity_3']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [224, 224, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'model_1/model/conv_handedness/MatMul;model_1/model/conv_handedness/BiasAdd', 'shape': [1, 1, 1, 1]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'model_1/model/conv_handflag/MatMul;model_1/model/conv_handflag/BiasAdd', 'shape': [1, 1, 1, 1]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'Identity', 'shape': [1, 1, 1, 63]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'Identity_3', 'shape': [1, 1, 1, 63]}
            inputs = "input_1"
            outputs = "Identity,model_1/model/conv_handedness/MatMul;model_1/model/conv_handedness/BiasAdd,model_1/model/conv_handflag/MatMul;model_1/model/conv_handflag/BiasAdd,Identity_3"
        elif model_name == "face_detection_short_range":
            assert (model_size==128), "face_detection_short_range resolution should be 128"
            #start_node_names = ['input']
            #end_node_names = ['regressor_16', 'regressor_8', 'classificator_16', 'classificator_8']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input', 'shape': [128, 128, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'classificator_8', 'shape': [16, 16, 1, 2]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'classificator_16', 'shape': [8, 8, 1, 6]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'regressor_8', 'shape': [16, 16, 1, 32]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'regressor_16', 'shape': [8, 8, 1, 96]}            
            inputs = "input"
            outputs = "regressor_16,regressor_8,classificator_16,classificator_8"
        elif model_name == "face_detection_full_range":
            assert (model_size==192), "face_detection_full_range resolution should be 192"
            #start_node_names = ['input']
            #end_node_names = ['regressor_face_4', 'classifier_face_4']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input', 'shape': [192, 192, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'classifier_face_4', 'shape': [48, 48, 1, 1]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'regressor_face_4', 'shape': [48, 48, 1, 16]}            
            inputs = "input"
            outputs = "regressor_face_4,classifier_face_4"
        elif model_name == "face_landmark":
            assert (model_size==192), "face_landmark resolution should be 192"
            #start_node_names = ['input_1']
            #end_node_names = ['conv2d_31','conv2d_21']
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [192, 192, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'conv2d_31', 'shape': [1, 1, 1, 1]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'conv2d_21', 'shape': [1, 1, 1, 1404]}            
            inputs = "input_1"
            outputs = "conv2d_31,conv2d_21"
        elif model_name == "pose_detection":
            assert (model_size==224), "pose_detection resolution should be 224"
            #start_node_names = ['input_1']
            #end_node_names = [
            #    'model_1/model/classifier_person_32_NO_PRUNING/BiasAdd;model_1/model/classifier_person_32_NO_PRUNING/Conv2D;model_1/model/classifier_person_32_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/classifier_person_16_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/classifier_person_8_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/regressor_person_32_NO_PRUNING/BiasAdd;model_1/model/regressor_person_32_NO_PRUNING/Conv2D1', 
            #    'model_1/model/regressor_person_16_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/regressor_person_8_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1'
            #]            
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [224, 224, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'model_1/model/classifier_person_8_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [28, 28, 1, 2]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'model_1/model/regressor_person_8_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [28, 28, 1, 24]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'model_1/model/classifier_person_16_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [14, 14, 1, 2]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'model_1/model/regressor_person_16_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [14, 14, 1, 24]}
            #MPU 3 output port 4: {'model_index': 0, 'layer_name': 'model_1/model/classifier_person_32_NO_PRUNING/BiasAdd;model_1/model/classifier_person_32_NO_PRUNING/Conv2D;model_1/model/classifier_person_32_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 'shape': [7, 7, 1, 6]}
            #MPU 3 output port 5: {'model_index': 0, 'layer_name': 'model_1/model/regressor_person_32_NO_PRUNING/BiasAdd;model_1/model/regressor_person_32_NO_PRUNING/Conv2D1', 'shape': [7, 7, 1, 72]}
            inputs = "input_1"
            outputs = "model_1/model/classifier_person_32_NO_PRUNING/BiasAdd;model_1/model/classifier_person_32_NO_PRUNING/Conv2D;model_1/model/classifier_person_32_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/classifier_person_16_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/classifier_person_8_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/regressor_person_32_NO_PRUNING/BiasAdd;model_1/model/regressor_person_32_NO_PRUNING/Conv2D1,model_1/model/regressor_person_16_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1,model_1/model/regressor_person_8_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1"
        elif model_name == "pose_landmark_lite" or model_name == "pose_landmark_full" or model_name == "pose_landmark_heavy":
            assert (model_size==256), "pose_landmark_* resolution should be 256"
            #start_node_names = ['input_1']
            #end_node_names = [
            #    'model_1/model/convld_3d/BiasAdd;model_1/model/convld_3d/Conv2D;model_1/model/convld_3d/BiasAdd/ReadVariableOp/resource1', 
            #    'model_1/model/activation_poseflag/Sigmoid',
            #    'Identity_2',
            #    'Identity_3',
            #    'model_1/model/convworld_3d/BiasAdd;model_1/model/convworld_3d/Conv2D;model_1/model/convworld_3d/BiasAdd/ReadVariableOp/resource1', 
            #]                
            #
            #MPU 0 input port 0: {'model_index': 0, 'layer_name': 'input_1', 'shape': [256, 256, 1, 3]}
            #MPU 3 output port 0: {'model_index': 0, 'layer_name': 'Identity_2', 'shape': [256, 256, 1, 1]}
            #MPU 3 output port 1: {'model_index': 0, 'layer_name': 'Identity_3', 'shape': [64, 64, 1, 39]}
            #MPU 3 output port 2: {'model_index': 0, 'layer_name': 'model_1/model/convld_3d/BiasAdd;model_1/model/convld_3d/Conv2D;model_1/model/convld_3d/BiasAdd/ReadVariableOp/resource1', 'shape': [1, 1, 1, 195]}
            #MPU 3 output port 3: {'model_index': 0, 'layer_name': 'model_1/model/convworld_3d/BiasAdd;model_1/model/convworld_3d/Conv2D;model_1/model/convworld_3d/BiasAdd/ReadVariableOp/resource1', 'shape': [1, 1, 1, 117]}
            #MPU 3 output port 4: {'model_index': 0, 'layer_name': 'model_1/model/conv_poseflag/BiasAdd;model_1/model/conv_poseflag/Conv2D;model_1/model/conv_poseflag/BiasAdd/ReadVariableOp/resource1', 'shape': [1, 1, 1, 1]}            
            inputs = "input_1"
            outputs = "model_1/model/convld_3d/BiasAdd;model_1/model/convld_3d/Conv2D;model_1/model/convld_3d/BiasAdd/ReadVariableOp/resource1,model_1/model/conv_poseflag/BiasAdd;model_1/model/conv_poseflag/Conv2D;model_1/model/conv_poseflag/BiasAdd/ReadVariableOp/resource1,Identity_2,Identity_3,model_1/model/convworld_3d/BiasAdd;model_1/model/convworld_3d/Conv2D;model_1/model/convworld_3d/BiasAdd/ReadVariableOp/resource1"
        else:
            inputs = ""
            outputs = ""

        print("[INFO] inputs : ",inputs)
        print("[INFO] outputs : ",outputs)
        
    else:
        print("[ERROR] Unknown model type : ",model_type)
        
    nc = NeuralCompiler(num_chips=4, models=model_path, verbose=1, dfp_fname=model_name, effort="Hard", show_optimization=True, autocrop=False, inputs=inputs, outputs=outputs )
    dfp = nc.run()




