# Used to convert original Mediapipe tflite models to ONNX
# https://github.com/PINTO0309/tflite2tensorflow
#
# First run: docker pull ghcr.io/pinto0309/tflite2tensorflow:latest
    
# Then run: docker pull ghcr.io/pinto0309/tflite2tensorflow
docker run -it --rm \
    -v `pwd`:/home/user/workdir \
    -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
    --device /dev/video0:/dev/video0:mwr \
    --net=host \
    -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    -e DISPLAY=$DISPLAY \
    --privileged \
    ghcr.io/pinto0309/tflite2tensorflow:latest bash    
