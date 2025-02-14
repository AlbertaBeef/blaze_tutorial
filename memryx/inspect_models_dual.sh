# TFLite models
#model_palm_detector_v0_07=("palm_detection_v0_07","models/palm_detection_v0_07.tflite",256)
#  UnsupportedOperationError in op conv2d_transpose: CUSTOM operation is unsupported
models_hand_v0_07=("palm_detection_v0_07","models/palm_detection_without_custom_op.tflite",256, "hand_landmark_v0_07","models/hand_landmark_v0_07.tflite",256, "hand_v0_07")

models_hand_v0_10_lite=("palm_detection_lite","models/palm_detection_lite.tflite",192, "hand_landmark_lite","models/hand_landmark_lite.tflite",224, "hand_v0_10_lite")
models_hand_v0_10_full=("palm_detection_full","models/palm_detection_full.tflite",192, "hand_landmark_full","models/hand_landmark_full.tflite",224, "hand_v0_10_full")

models_face_v0_10_short=("face_detection_short_range","models/face_detection_short_range.tflite",128, "face_landmark","models/face_landmark.tflite",192, "face_v0_10_short")
models_face_v0_10_full=( "face_detection_full_range" ,"models/face_detection_full_range.tflite" ,192, "face_landmark","models/face_landmark.tflite",192, "face_v0_10_full")

models_pose_v0_10_lite=( "pose_detection","models/pose_detection.tflite",224, "pose_landmark_lite" ,"models/pose_landmark_lite.tflite" ,256, "pose_v0_10_lite" )
models_pose_v0_10_full=( "pose_detection","models/pose_detection.tflite",224, "pose_landmark_full" ,"models/pose_landmark_full.tflite" ,256, "pose_v0_10_full" )
models_pose_v0_10_heavy=("pose_detection","models/pose_detection.tflite",224, "pose_landmark_heavy","models/pose_landmark_heavy.tflite",256, "pose_v0_10_heavy")

models_list=(
	models_hand_v0_07[@]
	models_hand_v0_10_lite[@]
	models_hand_v0_10_full[@]
	models_face_v0_10_short[@]
	models_face_v0_10_full[@]
	models_pose_v0_10_lite[@]
	models_pose_v0_10_full[@]
	models_pose_v0_10_heavy[@]		
)

# these only work when reverting tensorflow back to 2.16.2 (requires modified venv)
models_list=(
	models_pose_v0_10_lite[@]
	models_pose_v0_10_full[@]
	models_pose_v0_10_heavy[@]		
)


# these models worked
models_list=(
	models_hand_v0_07[@]
	models_face_v0_10_short[@]
	models_face_v0_10_full[@]
)

# these only worked when hand model is specified before palm model (handled in memryx_flow_dual.py script)
models_list=(
	models_hand_v0_10_lite[@]
	models_hand_v0_10_full[@]
)


models_count=${#models_list[@]}
#echo $models_count


# Convert to TensorFlow-Keras

for ((i=0; i<$models_count; i++))
do
	models=${!models_list[i]}
	models_array=(${models//,/ })
	model1_name=${models_array[0]}
	model1_file=${models_array[1]}
	model1_size=${models_array[2]}
	model2_name=${models_array[3]}
	model2_file=${models_array[4]}
	model2_size=${models_array[5]}
	dfp_name=${models_array[6]}

	echo python3 memryx_flow_dual.py --arch mx3 --dfp ${dfp_name} --name1 ${model1_name} --model1 ${model1_file} --resolution1 ${model1_size} --name2 ${model2_name} --model2 ${model2_file} --resolution2 ${model2_size} --process inspect

	python3 memryx_flow_dual.py --arch mx3 --dfp ${dfp_name} --name1 ${model1_name} --model1 ${model1_file} --resolution1 ${model1_size} --name2 ${model2_name} --model2 ${model2_file} --resolution2 ${model2_size} --process inspect | tee inspect_${dfp_name}.log

done
