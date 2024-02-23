import numpy as np
import cv2
import argparse
import os

from hailo_sdk_client import ClientRunner

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--arch'       , type=str,  default="hailo8", help="Hailo HW architecture.  Default is 'hailo8'.")
ap.add_argument('-b', '--blaze'      , type=str,  default="hand", help="Blaze application ('hand', 'face', 'pose').  Default is 'hand'")
ap.add_argument('-n', '--name'       , type=str,  default="palm_detection_lite", help="Model name. Default is 'palm_detection_lite'")
ap.add_argument('-m', '--model'      , type=str,  default="models/palm_detection_lite/model_float32.onnx", help="Model file (ONNX format). Default is 'models/palm_detection_lite/model_float32.onnx'")
ap.add_argument('-r', '--resolution' , type=int,  default=192, help="Input resolution.  Default is 192 for 192x192.")
ap.add_argument('-p', '--process'    , type=str,  default="all", help="Command seperated list of processes to run ( 'inspect', 'parse,optimize,compile' ). Default is 'all'")


args = ap.parse_args()  
  
print('Command line options:')
print(' --arch        : ', args.arch)
print(' --blaze       : ', args.blaze)
print(' --name        : ', args.name)
print(' --model       : ', args.model)
print(' --resolution  : ', args.resolution)
print(' --process     : ', args.process)


#
# Parsing
# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_1_Parsing_Tutorial.html
#

if ("inspect" in args.process):

    runner = ClientRunner(hw_arch=args.arch)
    model_path = args.model
    model_name = args.name

    hn, npz = runner.translate_onnx_model(model_path,model_name)

if ("parse" in args.process or args.process == "all"):

    runner = ClientRunner(hw_arch=args.arch)
    model_path = args.model
    model_name = args.name

    if args.name == "palm_detection_v0_07":
        start_node_names = ['input']
        assert (args.resolution==256), "palm_detection_v0_07 resolution should be 256"
        net_input_shapes={'input': [1, 3, args.resolution, args.resolution]}
        #end_node_names = ['regressors','classificators']
        end_node_names = ['Conv__533', 'Conv__544', 'Conv__551', 'Conv__532', 'Conv__543', 'Conv__550']
        # Conv__533 [1x6x8x8]   =transpose=> [1x8x8x6]   =reshape=> [1x384x1]  \\
        # Conv__544 [1x2x16x16] =transpose=> [1x16x16x2] =reshape=> [1x512x1]    => [1x2944x1]
        # Conv__551 [1x2x32x32] =transpose=> [1x32x23x2] =reshape=> [1x2048x1]  //
        #
        # Conv__532 [1x108x8x8]  =transpose=> [1x8x8x108]  =reshape=> [1x384x18]   \\
        # Conv__543 [1x36x16x16] =transpose=> [1x16x16x36] =reshape=> [1x512x18]    => [1x2944x18]
        # Conv__550 [1x36x32x32] =transpose=> [1x32x32x36] =reshape=> [1x2048x1]  //
    elif args.name == "hand_landmark_v0_07":
        start_node_names = ['input_1']
        assert (args.resolution==256), "hand_landmark_v0_07 resolution should be 256"
        net_input_shapes={'input_1': [1, 3, args.resolution, args.resolution]}
        end_node_names = ['ld_21_3d','output_handflag','output_handedness']
    elif args.name == "palm_detection_lite" or args.name == "palm_detection_full":
        start_node_names = ['input_1']
        assert (args.resolution==192), "palm_detection_lite/full resolution should be 192"
        net_input_shapes={'input_1': [1, 3, args.resolution, args.resolution]}
        #end_node_names = ['Identity','Identity_1']
        end_node_names = ['Conv__410', 'Conv__412','Conv__409','Conv__411']
        # Conv__410 [1x2x24x24] =transpose=> [1x24x24x2] =reshape=> [1x1152x1] \\
        #                                                                        => [1x2016x1]
        # Conv__412 [1x6x12x12] =transpose=> [1x12x12x6] =reshape=> [1x864x1]  //
        #
        # Conv__409 [1x36x24x24]  =transpose=> [1x24x24x36]  =reshape=> [1x1152x18] \\
        #                                                                             => [1x2016x18]
        # Conv__411 [1x108x12x12] =transpose=> [1x12x12x108] =reshape=> [1x864x18]  //
    elif args.name == "hand_landmark_lite" or args.name == "hand_landmark_full":
        start_node_names = ['input_1']
        assert (args.resolution==224), "hand_landmark_lite/full resolution should be 192"
        net_input_shapes={'input_1': [1, 3, args.resolution, args.resolution]}
        end_node_names = ['Identity','Identity_1','Identity_2','Identity_3']
    else:
        start_node_names = []
        end_node_names = []

   
    hn, npz = runner.translate_onnx_model(
        model_path, 
        model_name, 
        start_node_names=start_node_names,
        net_input_shapes=net_input_shapes,
        end_node_names=end_node_names)

    hailo_model_har_name = f'{model_name}_hailo_model.har'
    runner.save_har(hailo_model_har_name)

#
# Optimization
# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_2_Model_Optimization_Tutorial.html
#

if ("optimize" in args.process or args.process == "all"):

    model_name = args.name

    calib_dataset_file = "calib_dataset_"+str(args.resolution)+"x"+str(args.resolution)+".npy"
    calib_dataset = np.load(calib_dataset_file)

    hailo_model_har_name = f'{model_name}_hailo_model.har'
    assert os.path.isfile(hailo_model_har_name), 'Please provide valid path for HAR file'
    runner = ClientRunner(har=hailo_model_har_name)
    # By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.

    # Now we will create a model script, that tells the compiler to add a normalization on the beginning
    # of the model (that is why we didn't normalize the calibration set;
    # Otherwise we would have to normalize it before using it)

    # Batch size is 8 by default
    #alls = 'normalization1 = normalization([255.0 255.0 255.0], [0.0 0.0 0.0])\n'

    # Load the model script to ClientRunner so it will be considered on optimization
    #runner.load_model_script(alls)

    # Call Optimize to perform the optimization process
    runner.optimize(calib_dataset)

    # Save the result state to a Quantized HAR file
    quantized_model_har_path = f'{model_name}_quantized_model.har'
    runner.save_har(quantized_model_har_path)

#
# Compilation
# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_3_Compilation_Tutorial.html
#

if ("compile" in args.process or args.process == "all"):

    model_name = args.name

    quantized_model_har_path = f'{model_name}_quantized_model.har'

    runner = ClientRunner(har=quantized_model_har_path)
    # By default it uses the hw_arch that is saved on the HAR. It is not recommended to change the hw_arch after Optimization.

    hef = runner.compile()

    file_name = f'{model_name}.hef'
    with open(file_name, 'wb') as f:
        f.write(hef)
    
    har_path = f'{model_name}_compiled_model.har'
    runner.save_har(har_path)

    #!hailo profiler {har_path}    


