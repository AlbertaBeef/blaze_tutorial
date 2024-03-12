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
ap.add_argument('-p', '--process'    , type=str,  default="all", help="Command seperated list of processes to run ( 'inspect', 'profile', 'all'=='parse,optimize,compile' ). Default is 'all'")


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
    
    file_name, model_type = os.path.splitext(model_path)
    if model_type == ".tflite":
        hn, npz = runner.translate_tf_model(model_path,model_name)
    elif model_type == ".onnx":
        hn, npz = runner.translate_onnx_model(model_path,model_name)
    else:
        print("[ERROR] Unknown model type : ",model_type)

if ("parse" in args.process or "profile" in args.process or args.process == "all"):

    runner = ClientRunner(hw_arch=args.arch)
    model_path = args.model
    model_name = args.name

    file_name, model_type = os.path.splitext(model_path)
    if model_type == ".tflite":
    
        if args.name == "palm_detection_v0_07":
            assert (args.resolution==256), "palm_detection_v0_07 resolution should be 256"
            start_node_names = ['input']
            #end_node_names = ['classificators','regressors']
            #end_node_names = ['classificator_8','classificator_16', 'classificator_32','regressor_8','regressor_16', 'regressor_32']
            end_node_names = ['activation_41', 'activation_23', 'activation_31', 'activation_39']
        elif args.name == "hand_landmark_v0_07":
            assert (args.resolution==256), "hand_landmark_v0_07 resolution should be 256"
            start_node_names = ['input_1']
            end_node_names = ['ld_21_3d','output_handflag','output_handedness']
        elif args.name == "palm_detection_lite" or args.name == "palm_detection_full":
            assert (args.resolution==192), "palm_detection_lite/full resolution should be 192"
            start_node_names = ['input_1']
            #end_node_names = ['Identity','Identity_1']
            #hailo_sdk_common.hailo_nn.exceptions.UnsupportedModelError: 1D form is not supported in layer Identity_1 of type ConcatLayer.
            #end_node_names = ['model_1/model/reshaped_classifier_palm_16/Reshape', 'model_1/model/reshaped_classifier_palm_8/Reshape','model_1/model/reshaped_regressor_palm_16/Reshape','model_1/model/reshaped_regressor_palm_8/Reshape']
            #hailo_sdk_client.model_translator.exceptions.ParsingWithRecommendationException: Parsing failed. The errors found in the graph are:
            # UnsupportedShuffleLayerError in op model_1/model/reshaped_classifier_palm_8/Reshape: Failed to determine type of layer to create in node model_1/model/reshaped_classifier_palm_8/Reshape (RESHAPE)
            # UnsupportedShuffleLayerError in op model_1/model/reshaped_regressor_palm_8/Reshape: Failed to determine type of layer to create in node model_1/model/reshaped_regressor_palm_8/Reshape (RESHAPE)
            # UnsupportedShuffleLayerError in op model_1/model/reshaped_classifier_palm_16/Reshape: Failed to determine type of layer to create in node model_1/model/reshaped_classifier_palm_16/Reshape (RESHAPE)
            # UnsupportedShuffleLayerError in op model_1/model/reshaped_regressor_palm_16/Reshape: Failed to determine type of layer to create in node model_1/model/reshaped_regressor_palm_16/Reshape (RESHAPE)
            #Please try to parse the model again, using these end node names: model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1, model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1, model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1, model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1
            end_node_names = [
                'model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_16_NO_PRUNING/Conv2D;model_1/model/classifier_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd;model_1/model/classifier_palm_8_NO_PRUNING/Conv2D;model_1/model/classifier_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_16_NO_PRUNING/Conv2D;model_1/model/regressor_palm_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd;model_1/model/regressor_palm_8_NO_PRUNING/Conv2D;model_1/model/regressor_palm_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1']
        elif args.name == "hand_landmark_lite" or args.name == "hand_landmark_full":
            assert (args.resolution==224), "hand_landmark_lite/full resolution should be 192"
            start_node_names = ['input_1']
            end_node_names = ['Identity','Identity_1','Identity_2','Identity_3']
        elif args.name == "face_detection_short_range":
            assert (args.resolution==128), "face_detection_short_range resolution should be 128"
            start_node_names = ['input']
            #end_node_names = ['regressors', 'classificators']
            #end_node_names = ['reshape','reshape_2','reshape_1','reshape_3']
            end_node_names = ['regressor_16', 'regressor_8', 'classificator_16', 'classificator_8']
        elif args.name == "face_detection_full_range":
            assert (args.resolution==192), "face_detection_full_range resolution should be 192"
            start_node_names = ['input']
            #end_node_names = ['reshaped_regressor_face_4', 'reshaped_classifier_face_4']
            end_node_names = ['regressor_face_4', 'classifier_face_4']
        elif args.name == "face_landmark":
            assert (args.resolution==192), "face_landmark resolution should be 192"
            start_node_names = ['input_1']
            end_node_names = ['conv2d_31','conv2d_21']
        elif args.name == "pose_detection":
            assert (args.resolution==224), "pose_detection resolution should be 224"
            start_node_names = ['input_1']
            end_node_names = ['Identity','Identity_1']
            # ValueError: cannot reshape array of size 96 into shape (16,1,1,24)
            #end_node_names = ['model_1/model/reshaped_classifier_person_8/Reshape', 'model_1/model/reshaped_classifier_person_16/Reshape','model_1/model/reshaped_classifier_person_32/Reshape','model_1/model/reshaped_regressor_person_8/Reshape','model_1/model/reshaped_regressor_person_16/Reshape','model_1/model/reshaped_regressor_person_32/Reshape']
            '''
            end_node_names = [
                'model_1/model/classifier_person_32_NO_PRUNING/BiasAdd;model_1/model/classifier_person_32_NO_PRUNING/Conv2D;model_1/model/classifier_person_32_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/classifier_person_16_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/classifier_person_8_NO_PRUNING/BiasAdd;model_1/model/classifier_person_16_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/Conv2D;model_1/model/classifier_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/regressor_person_32_NO_PRUNING/BiasAdd;model_1/model/regressor_person_32_NO_PRUNING/Conv2D1', 
                'model_1/model/regressor_person_16_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_16_NO_PRUNING/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/regressor_person_8_NO_PRUNING/BiasAdd;model_1/model/regressor_person_16_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/Conv2D;model_1/model/regressor_person_8_NO_PRUNING/BiasAdd/ReadVariableOp/resource1'
            ]            
            '''
        elif args.name == "pose_landmark_lite" or args.name == "pose_landmark_full" or args.name == "pose_landmark_heavy":
            assert (args.resolution==256), "pose_landmark_* resolution should be 256"
            start_node_names = ['input_1']
            #end_node_names = ['Identity','Identity_1','Identity_2','Identity_3','Identity_4']
            end_node_names = [
                'model_1/model/convld_3d/BiasAdd;model_1/model/convld_3d/Conv2D;model_1/model/convld_3d/BiasAdd/ReadVariableOp/resource1', 
                'model_1/model/activation_poseflag/Sigmoid',
                'Identity_2',
                'Identity_3',
                'model_1/model/convworld_3d/BiasAdd;model_1/model/convworld_3d/Conv2D;model_1/model/convworld_3d/BiasAdd/ReadVariableOp/resource1', 
            ]                
        else:
            start_node_names = []
            end_node_names = []

        print("[INFO] start_node_names : ",start_node_names)
        print("[INFO] end_node_names : ",end_node_names)
        
        hn, npz = runner.translate_tf_model(
            model_path, 
            model_name, 
            start_node_names=start_node_names,
            end_node_names=end_node_names)
    
    elif model_type == ".onnx":

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
            #end_node_names = ['Identity','Identity_1','Identity_2','Identity_3']
            #[VStreamInfo("hand_landmark_lite/fc1"), VStreamInfo("hand_landmark_lite/fc4"), VStreamInfo("hand_landmark_lite/fc3"), VStreamInfo("hand_landmark_lite/fc2")]
            end_node_names = ['Identity','Identity_2','Identity_3','Identity_1']
            #[VStreamInfo("hand_landmark_lite/fc1"), VStreamInfo("hand_landmark_lite/fc4"), VStreamInfo("hand_landmark_lite/fc3"), VStreamInfo("hand_landmark_lite/fc2")]            
        else:
            start_node_names = []
            end_node_names = []

        hn, npz = runner.translate_onnx_model(
            model_path, 
            model_name, 
            start_node_names=start_node_names,
            net_input_shapes=net_input_shapes,
            end_node_names=end_node_names)

    else:
        print("[ERROR] Unknown model type : ",model_type)
        
    hailo_model_har_name = f'{model_name}_hailo_model.har'
    runner.save_har(hailo_model_har_name)

#
# Profile
# Reference: ...
#

if ("profile" in args.process):

   model_name = args.name
   # hn, npz ... from previous step ...
   
   detailed, summary = runner.profile_hn_model(fps=120)   


#
# Optimization
# Reference : https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-26-0/?sp_referrer=tutorials_notebooks/notebooks/DFC_2_Model_Optimization_Tutorial.html
#

if ("optimize" in args.process or args.process == "all"):

    model_name = args.name

    hailo_model_har_name = f'{model_name}_hailo_model.har'
    assert os.path.isfile(hailo_model_har_name), 'Please provide valid path for HAR file'
    runner = ClientRunner(har=hailo_model_har_name)
    # By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.

    # Now we will create a model script, that tells the compiler to add a normalization on the beginning
    # of the model (that is why we didn't normalize the calibration set;
    # Otherwise we would have to normalize it before using it)

    # Batch size is 8 by default
    #alls = 'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n'

    if args.name == "palm_detection_lite" or args.name == "palm_detection_full":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        #calib_dataset_file = "calib_hand_dataset_"+str(args.resolution)+"x"+str(args.resolution)+".npy"
        calib_dataset_file = "calib_palm_detection_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)


    if args.name == "hand_landmark_lite" or args.name == "hand_landmark_full":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        #calib_dataset_file = "calib_hand_dataset_"+str(args.resolution)+"x"+str(args.resolution)+".npy"
        calib_dataset_file = "calib_hand_landmark_224_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if args.name == "face_detection_short_range":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        calib_dataset_file = "calib_face_detection_128_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if args.name == "face_detection_full_range":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        calib_dataset_file = "calib_face_detection_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if args.name == "face_landmark":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        calib_dataset_file = "calib_face_landmark_192_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if args.name == "pose_detection":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        calib_dataset_file = "calib_pose_detection_224_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)

    if args.name == "pose_landmark_lite" or args.name == "pose_landmark_full" or args.name == "pose_landmark_heavy":
        alls = 'input_normalization = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'

        # Load the model script to ClientRunner so it will be considered on optimization
        runner.load_model_script(alls)

        # Specify calibration dataset
        calib_dataset_file = "calib_pose_landmark_256_dataset.npy"
        calib_dataset = np.load(calib_dataset_file)
    
    # Randomize calibration dataset
    #print("[INFO] calib_dataset_file : ",calib_dataset_file )
    #print("[INFO] calib_dataset (before shuffle) : ",calib_dataset.shape )
    calib_dataset = np.take(calib_dataset,np.random.permutation(calib_dataset.shape[0]),axis=0,out=calib_dataset)
    #print("[INFO] calib_dataset (after  shuffle) : ",calib_dataset.shape )
        
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


