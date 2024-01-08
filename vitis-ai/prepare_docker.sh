#/bin/bash

# req:
#   docker pull xilinx/vitis-ai-pytorch-cpu
#   docker pull xilinx/vitis-ai-pytorch-rocm
# usage:
#   ./docker_run.sh xilinx/vitis-ai-pytorch-cpu
#   ./docker_run.sh xilinx/vitis-ai-pytorch-rocm

wget https://github.com/Xilinx/Vitis-AI/raw/v3.5/docker_run.sh
chmod a+x docker_run.sh

wget https://github.com/Xilinx/Vitis-AI/raw/v3.5/docker/dockerfiles/PROMPT/PROMPT_cpu.txt
wget https://github.com/Xilinx/Vitis-AI/raw/v3.5/docker/dockerfiles/PROMPT/PROMPT_gpu.txt
wget https://github.com/Xilinx/Vitis-AI/raw/v3.5/docker/dockerfiles/PROMPT/PROMPT_rocm.txt

mkdir -p docker/dockerfiles/PROMPT
mv PROMPT_cpu.txt docker/dockerfiles/PROMPT/.
mv PROMPT_gpu.txt docker/dockerfiles/PROMPT/.
mv PROMPT_rocm.txt docker/dockerfiles/PROMPT/.
