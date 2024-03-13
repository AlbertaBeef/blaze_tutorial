# PyTorch models
model_palm_detector_v0_07=("BlazePalm",         256)
model_hand_landmark_v0_07=("BlazeHandLandmark", 256)
model_face_detector_v0_07_front=("BlazeFace",   128)
model_face_detector_v0_07_back=("BlazeFaceBack",256)
model_face_landmark_v0_07=("BlazeFaceLandmark", 192)
model_pose_detector_v0_07=("BlazePose",         128)
model_pose_landmark_v0_07=("BlazePoseLandmark", 256)

model_list=(
    model_palm_detector_v0_07[@]
    model_hand_landmark_v0_07[@]
    model_face_detector_v0_07_back[@]
    model_face_detector_v0_07_front[@]
    model_face_landmark_v0_07[@]
    model_pose_detector_v0_07[@]
    model_pose_landmark_v0_07[@]
)
model_list=(
    model_face_detector_v0_07_back[@]
    model_face_detector_v0_07_front[@]
    model_face_landmark_v0_07[@]
)

# Versal AI Edge
dpu_c20b14=("C20B14","./arch/C20B14/arch-c20b14.json")
dpu_c20b1=("C20B1","./arch/C20B14/arch-c20b1.json")
# Zynq-UltraScale+
dpu_b4096=("B4096","./arch/B4096/arch-zcu104.json")
dpu_b3136=("B3136","./arch/B3136/arch-kv260.json")
dpu_b2304=("B2304","./arch/B2304/arch-b2304-lr.json")
dpu_b1152=("B1152","./arch/B1152/arch-b1152-hr.json")
dpu_b512=("B512","./arch/B512/arch-b512-lr.json")
dpu_b128=("B128","./arch/B128/arch-b128-lr.json")
#
dpu_arch_list=(
    dpu_c20b14[@]
    dpu_c20b1[@]
    dpu_b4096[@]
    dpu_b3136[@]
    dpu_b2304[@]
    dpu_b1152[@]
    dpu_b512[@]
    dpu_b128[@]
)

model_count=${#model_list[@]}
#echo $model_count

dpu_arch_count=${#dpu_arch_list[@]}
#echo $dpu_arch_count


# Model

for ((i=0; i<$model_count; i++))
do
    model=${!model_list[i]}
    model_array=(${model//,/ })
    model_name=${model_array[0]}
    input_resolution=${model_array[1]}

    echo python3 vitisai_pytorch_flow.py --name ${model_name} --resolution ${input_resolution} --process all
    python3 vitisai_pytorch_flow.py --name ${model_name} --resolution ${input_resolution} --process all | tee deploy_${model_name}_quantize.log

    if [ ${model_name} == "BlazeFaceBack" ] 
    then
        mv quantize_result/BlazeFace_int.pt     quantize_result/BlazeFaceBack_int.pt
        mv quantize_result/BlazeFace_int.onnx   quantize_result/BlazeFaceBack_int.onnx
        mv quantize_result/BlazeFace_int.xmodel quantize_result/BlazeFaceBack_int.xmodel
    fi
        
    for ((j=0; j<$dpu_arch_count; j++))
    do  
        dpu=${!dpu_arch_list[j]}
        dpu_array=(${dpu//,/ })
        dpu_arch=${dpu_array[0]}
        dpu_json=${dpu_array[1]}
        
        echo vai_c_xir -x ./quantize_result/${model_name}_int.xmodel -a ${dpu_json} -o ./models_blaze/${model_name}/${dpu_arch} -n ${model_name}
        vai_c_xir -x ./quantize_result/${model_name}_int.xmodel -a ${dpu_json} -o ./models_blaze/${model_name}/${dpu_arch} -n ${model_name} | tee deploy_${model_name}_compile.log
    done
done
