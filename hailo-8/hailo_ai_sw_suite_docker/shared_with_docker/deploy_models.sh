# ONNX models
model_palm_detector_v0_07=("palm_detection_v0_07","models/palm_detection_v0_07/model_float32.onnx",256)
model_palm_detector_v0_10_lite=("palm_detection_lite","models/palm_detection_lite/model_float32.onnx",192)
model_palm_detector_v0_10_full=("palm_detection_full","models/palm_detection_full/model_float32.onnx",192)
model_hand_landmark_v0_07=("hand_landmark_v0_07","models/hand_landmark_v0_07/model_float32.onnx",256)
model_hand_landmark_v0_10_lite=("hand_landmark_lite","models/hand_landmark_lite/model_float32.onnx",224)
model_hand_landmark_v0_10_full=("hand_landmark_full","models/hand_landmark_full/model_float32.onnx",224)
model_list=(
	model_palm_detector_v0_07[@]
	model_palm_detector_v0_10_lite[@]
	model_palm_detector_v0_10_full[@]
	model_hand_landmark_v0_07[@]
	model_hand_landmark_v0_10_lite[@]
	model_hand_landmark_v0_10_full[@]
)
# TFLite models
model_palm_detector_v0_10_lite=("palm_detection_lite","models/palm_detection_lite.tflite",192)
model_palm_detector_v0_10_full=("palm_detection_full","models/palm_detection_full.tflite",192)
model_hand_landmark_v0_10_lite=("hand_landmark_lite","models/hand_landmark_lite.tflite",224)
model_hand_landmark_v0_10_full=("hand_landmark_full","models/hand_landmark_full.tflite",224)
model_list=(
	model_palm_detector_v0_10_lite[@]
	model_palm_detector_v0_10_full[@]
	model_hand_landmark_v0_10_lite[@]
	model_hand_landmark_v0_10_full[@]
)
model_count=${#model_list[@]}
#echo $model_count


# Convert to TensorFlow-Keras

for ((i=0; i<$model_count; i++))
do
	model=${!model_list[i]}
	model_array=(${model//,/ })
	model_name=${model_array[0]}
	model_file=${model_array[1]}
	input_resolution=${model_array[2]}

	echo python3 hailo_flow.py --name ${model_name} --model ${model_file} --resolution ${input_resolution} --process all

	python3 hailo_flow.py --name ${model_name} --model ${model_file} --resolution ${input_resolution} --process all | tee deploy_${model_name}.log

done
