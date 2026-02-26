# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cross-platform hardware accelerator benchmark and tutorial for deploying Google MediaPipe Blaze models (face detection, hand tracking, pose estimation) on five embedded AI accelerator ecosystems. The same model family is quantized, compiled, and deployed across all targets. Copyright 2024 Avnet Inc., Apache 2.0 license.

Hackster guides: http://avnet.me/mediapipe-03-vitis-ai-3.5 (Vitis-AI), http://avnet.me/mediapipe-04-hailo-8 (Hailo-8).

## Branch Structure — Five Hardware Targets

Each branch adds an accelerator-specific flow. `mx3`, `qcs6490`, and `axelera` branch from `2023.1` HEAD.

| Branch | Accelerator | Key Directory | Toolchain |
|--------|------------|---------------|-----------|
| `main` | — | — | Base repo with MediaPipePyTorch submodule only |
| `2023.1` (default) | Xilinx DPU + Hailo-8 | `vitis-ai/`, `hailo-8/` | Vitis-AI 3.5 (`pytorch_nndct`, `vai_c_xir`), Hailo AI SW Suite Docker |
| `mx3` | MemryX MX3 | `memryx/` | `memryx_flow.py` / `memryx_flow_dual.py`, TFLite model download + MX3 compilation |
| `qcs6490` | Qualcomm QCS6490 | `qcs6490/` | Qualcomm AI Hub Workbench (`qai_hub_workbench_flow.py`), TFLite → `.dlc`/`.bin` |
| `axelera` | Axelera Metis | `axelera/` | Voyager SDK (`axelera_flow.py`), TFLite → ONNX → Metis AIPU compilation |

**Tags track milestones**: `vitis_ai_3.5_version_1`–`4` → `hailo8_version_2` → `hailo8l_rpi5_take1` → `mx_version_1` → `qcs6490_version_1`–`2`.

## Architecture

### Model Pipeline
MediaPipe models (PyTorch, via `MediaPipePyTorch/` submodule) → Quantization → Compilation → Hardware inference.

**Supported models**: BlazePalm, BlazeHandLandmark, BlazeFace, BlazeFaceBack, BlazeFaceLandmark, BlazePose, BlazePoseLandmark.

### Vitis-AI Flow (`vitis-ai/`)
- **`vitisai_pytorch_flow.py`** — Central quantization/compilation script using `pytorch_nndct`. Modes: `inspect`, `calib` (quantization calibration), `test`, `onnx`/`torchscript`/`xmodel` export.
- **`deploy_models.sh`** — Batch compilation across 9 DPU architectures (B128–B4096, C20B1–C20B14) using `vai_c_xir`.
- **`app/blazebase.py`** — Core inference engine. `BlazeDetector` (detection models) and `BlazeLandmark` (landmark models) inherit from `BlazeBase`. Loads XIR graphs via `xir` library for DPU execution.
- **`app/blaze*.py`** — Thin model-specific wrappers (anchors, config) around `blazebase.py`.
- **`app/blazepalm_detect_live.py`, `app/blazehandlandmark_detect_live.py`** — Live camera demos using v4l2.
- **`app/cpu_tasks/`** — Custom C++ DPU operators (`op_eltwise-fix/`, `op_pad-fix/`), built with CMake.
- **`app/arch/`** — JSON DPU architecture configs for each target.

### Hailo-8 Flow (`hailo-8/`) — branch `2023.1`
- Docker-based compilation via `hailo_ai_sw_suite_docker/`.
- TFLite-to-ONNX conversion in `tflite2onnx/` using PINTO0309/tflite2tensorflow Docker image.
- Supports both Hailo-8 and Hailo-8L (Raspberry Pi 5).

### MemryX MX3 Flow (`memryx/`) — branch `mx3`
- `memryx_flow.py` — Single-model compilation for MX3 accelerator.
- `memryx_flow_dual.py` — Dual-model compilation (e.g., detector + landmark model together).
- `deploy_models.sh` / `deploy_models_dual.sh` — Batch compilation scripts.
- `models/get_tflite_models.sh` — Downloads TFLite source models.

### Qualcomm QCS6490 Flow (`qcs6490/`) — branch `qcs6490`
- `qai_hub_workbench_flow.py` — Central compilation script using Qualcomm AI Hub Workbench. Converts models to `.dlc`/`.bin` for Qualcomm NPU.
- `deploy_models_qai_hub_workbench.sh` — Batch compilation script.
- `models/convert_models.sh` / `models/get_tflite_models.sh` — Model conversion and download.

### Axelera Metis Flow (`axelera/`) — branch `axelera`
- `axelera_flow.py` — Central quantization/compilation script using Axelera Voyager SDK. Quantizes ONNX models and compiles for Metis AIPU.
- `deploy_models.sh` — Batch compilation script.
- `models/convert_models.sh` / `models/get_tflite_models.sh` — TFLite download and ONNX conversion.

### Key Dependencies
Python: `torch`, `pytorch_nndct`, `cv2`, `numpy`, `xir`, `vitis_ai_library` (Vitis-AI); `memryx` SDK (MX3); `qai_hub` (Qualcomm); `axelera` Voyager SDK (Metis).
C++: CMake for custom operator builds in `vitis-ai/app/cpu_tasks/`.

## Common Commands

### Quantization (Vitis-AI, run inside Vitis-AI Docker)
```bash
cd vitis-ai
python vitisai_pytorch_flow.py --model blazepalm --mode calib    # calibration
python vitisai_pytorch_flow.py --model blazepalm --mode test     # test quantized model
python vitisai_pytorch_flow.py --model blazepalm --mode xmodel   # export XIR model
```

### Batch Compilation (all models × all DPU architectures)
```bash
cd vitis-ai
bash deploy_models.sh
```

### Build Custom CPU Operators
```bash
cd vitis-ai/app/cpu_tasks/op_eltwise-fix
mkdir build && cd build && cmake .. && make
```

### Live Inference (on target board with DPU)
```bash
cd vitis-ai/app
python blazepalm_detect_live.py
python blazehandlandmark_detect_live.py
```

### Hailo-8 Docker Workflow
```bash
cd hailo-8/hailo_ai_sw_suite_docker
bash hailo_ai_sw_suite_docker_run.sh
```

### MemryX MX3 (branch `mx3`)
```bash
git checkout mx3
cd memryx
bash models/get_tflite_models.sh                # download TFLite models
bash deploy_models.sh                           # single-model compilation
bash deploy_models_dual.sh                      # dual-model compilation
```

### Qualcomm QCS6490 (branch `qcs6490`)
```bash
git checkout qcs6490
cd qcs6490
bash models/get_tflite_models.sh                # download TFLite models
bash models/convert_models.sh                   # convert models
bash deploy_models_qai_hub_workbench.sh         # compile for Qualcomm NPU
```

### Axelera Metis (branch `axelera`)
```bash
git checkout axelera
cd axelera
bash models/get_tflite_models.sh                # download TFLite models
cd models && bash convert_models.sh && cd ..    # convert TFLite to ONNX
bash deploy_models.sh                           # quantize + compile for Metis AIPU
```

## Notes

- The `MediaPipePyTorch/` directory is a git submodule (zmurez/MediaPipePyTorch). Initialize with `git submodule update --init`.
- Jupyter notebooks in `vitis-ai/` (`blazepalm_exploration*.ipynb`, `blazehandlandmark_exploration*.ipynb`) are for development exploration, not formal tests.
- No formal test suite or CI/CD exists — validation is done via demo scripts and notebooks.
- DPU architecture JSON files in `app/arch/` must match the target hardware.
