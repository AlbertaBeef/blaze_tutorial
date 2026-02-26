# ONNX models
model_palm_detector_v0_07=("palm_detection_v0_07","models/palm_detection_v0_07.onnx",256)
model_hand_landmark_v0_07=("hand_landmark_v0_07","models/hand_landmark_v0_07.onnx",256)

model_palm_detector_v0_10_lite=("palm_detection_lite","models/palm_detection_lite.onnx",192)
model_palm_detector_v0_10_full=("palm_detection_full","models/palm_detection_full.onnx",192)
model_hand_landmark_v0_10_lite=("hand_landmark_lite","models/hand_landmark_lite.onnx",224)
model_hand_landmark_v0_10_full=("hand_landmark_full","models/hand_landmark_full.onnx",224)

model_face_detector_v0_10_short=("face_detection_short_range","models/face_detection_short_range.onnx",128)
model_face_detector_v0_10_full=("face_detection_full_range","models/face_detection_full_range.onnx",192)
model_face_landmark_v0_10=("face_landmark","models/face_landmark.onnx",192)

model_pose_detector_v0_10=("pose_detection","models/pose_detection.onnx",224)
model_pose_landmark_v0_10_lite=("pose_landmark_lite","models/pose_landmark_lite.onnx",256)
model_pose_landmark_v0_10_full=("pose_landmark_full","models/pose_landmark_full.onnx",256)
model_pose_landmark_v0_10_heavy=("pose_landmark_heavy","models/pose_landmark_heavy.onnx",256)

# start small ... one at a time ...
model_list=(
	model_palm_detector_v0_10_lite[@]
)

model_list=(
	model_palm_detector_v0_07[@]
	model_hand_landmark_v0_07[@]
	model_palm_detector_v0_10_full[@]
	model_hand_landmark_v0_10_full[@]
	model_palm_detector_v0_10_lite[@]
	model_hand_landmark_v0_10_lite[@]
	model_face_detector_v0_10_short[@]
	model_face_detector_v0_10_full[@]
	model_face_landmark_v0_10[@]
	model_pose_detector_v0_10[@]
	model_pose_landmark_v0_10_lite[@]
	model_pose_landmark_v0_10_full[@]
	model_pose_landmark_v0_10_heavy[@]
)

model_count=${#model_list[@]}
#echo $model_count


for ((i=0; i<$model_count; i++))
do
	model=${!model_list[i]}
	model_array=(${model//,/ })
	model_name=${model_array[0]}
	model_file=${model_array[1]}
	input_resolution=${model_array[2]}

	echo python3 axelera_flow.py --name ${model_name} --model ${model_file} --resolution ${input_resolution}

	python3 axelera_flow.py --name ${model_name} --model ${model_file} --resolution ${input_resolution} | tee deploy_${model_name}.log

done
