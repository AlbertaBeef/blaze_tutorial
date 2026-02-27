# Knowledge base for compiling models for the DeepX M1 accelerator (2026/02/27)

## DeepX M1: Model Compilation with DX-COM

This guide covers how to compile ONNX models for the DeepX M1 accelerator using the DX-COM tool, including example commands and scripts. It emphasizes single-input models (batch size fixed to 1) and provides practical scripts to streamline the process.
1) Overview

    DX-COM converts a pre-trained ONNX model and its JSON configuration into a hardware-optimized .dxnn binary for DEEPX NPUs, including the M1 accelerator.
    The ONNX file defines the model structure and weights; the JSON config defines pre/post-processing, calibration, and compilation parameters.
    Output: a single .dxnn file named after the ONNX model (e.g., MobilenetV1.dxnn).

Important notes:

    Models must be single-input (DX-COM currently does not support multi-input ONNX models).
    The batch size in the ONNX model should be set to 1 for compilation. If needed, resize/reshape the ONNX model prior to compiling (e.g., via onnxsim).
    Output may vary between runs due to internal optimization kernels. Use --shrink to produce a minimal and more deterministic output if reproducibility is important.

2) Prerequisites

    ONNX model file (single input, batch size 1)
    JSON configuration file for compilation
    DX-COM executable available (dx_com)
    Optional: onnxsim for batch-size-1 conditioning
    Environment with supported OS and dependencies per DX-COM requirements

3) Compile flow for the M1 accelerator

    Steps:
        Prepare ONNX model and JSON config
        If the ONNX model uses batch size > 1, convert to batch size 1 with onnxsim
        Run DX-COM to produce the .dxnn output
        (Optional) Use --shrink to minimize output size

    Important constraint:
        Ensure the ONNX input name and the config inputs match exactly.

4) Example commands

    Basic compile (single-input ONNX to DXNN)
        dx_com -m path/to/YourModel.onnx -c path/to/YourModel.json -o output/YourModel

    Compile with shrink (minimal output for NPU)
        dx_com -m path/to/YourModel.onnx -c path/to/YourModel.json -o output/YourModel --shrink

    Print version info (optional)
        dx_com -v
        dx_com -i

    Compile in a Dockerized environment (example)
        docker run --rm -v /host/workspace:/workspace -w /workspace deepx/dx-com:latest
        dx_com -m models/MobilenetV1.onnx -c models/MobilenetV1.json -o output/mobilenetv1

    Compile using a Makefile (example targets)
        Makefile example:
            mv1:
                dx_com -m sample/MobilenetV1.onnx -c sample/MobilenetV1.json -o output/mobilenetv1
            resnet50:
                dx_com -m sample/ResNet50.onnx -c sample/ResNet50.json -o output/ResNet50

    Overwrite batch size to 1 for an ONNX with dynamic or larger batch:
        If using onnxsim:
            pip install onnxsim
            python3 -m onnxsim YourModel.onnx YourModel_sim.onnx --overwrite-input-shape 1,3,224,224
            dx_com -m YourModel_sim.onnx -c YourModel.json -o output/YourModel_sim

    Example with a specific model (MobilenetV1)
        dx_com -m sample/MobilenetV1.onnx -c sample/MobilenetV1.json -o output/mobilenetv1

    Example with ResNet50
        dx_com -m sample/ResNet50.onnx -c sample/ResNet50.json -o output/ResNet50

    Example with YOLO (single-input variant)
        dx_com -m sample/YOLOV5-1.onnx -c sample/YOLOV5-1.json -o output/YOLOV5-1

Notes:

    If you encounter missing library or OS compatibility errors, verify the environment meets the DX-COM requirements (library versions, Ubuntu version, etc.).

5) Makefile-based automation (example)

    Sample Makefile (simplified)
        .PHONY: mv1 resnet50 yolov5
        mv1:
            dx_com -m modelzoo/onnx/MobilenetV1.onnx -c modelzoo/json/MobilenetV1.json -o output/mobilenetv1
        resnet50:
            dx_com -m modelzoo/onnx/ResNet50.onnx -c modelzoo/json/ResNet50.json -o output/ResNet50
        yolov5:
            dx_com -m modelzoo/onnx/YOLOV5.onnx -c modelzoo/json/YOLOV5.json -o output/YOLOV5

Usage:

    make mv1

    make resnet50

    make yolov5

    Shrink example in Makefile:
        dx_com ... -o output/Model --shrink

6) Batch size handling and pre/post processing

    Batch size in ONNX must be 1 for compilation.
    If your ONNX uses dynamic batch or batch > 1, convert to batch 1 using onnxsim:
        pip install onnxsim
        python3 -m onnxsim YourModel.onnx YourModel_sim.onnx --overwrite-input-shape 1,C,A,B
    Calibration and data loading (from JSON) should align with the model expectations.

7) Output structure

    After a successful compilation, you will typically find:
        {model_name}.dxnn at the chosen output directory
        Optional: calibration datasets and intermediate build files if not using --shrink
    Example:
        output/mobilenetv1/MobilenetV1.dxnn

8) Troubleshooting quick tips

    NotSupportError or NodeNotFoundError:
        Ensure the ONNX model uses supported operators and a single input.
    ConfigFileError/ConfigInputError:
        Verify the JSON syntax and that input names/shapes match the ONNX model.
    OnnxFileNotFound/DatasetPathError:
        Confirm file paths exist and are accessible from the build environment.
    Output size differs between runs:
        This can be due to internal kernel behavior. Consider using --shrink for deterministic outputs.

9) Quick-start checklist

    Prepare a single-input ONNX model (batch size 1) and config JSON
    If needed, convert to batch-1 ONNX with onnxsim
    Compile with DX-COM:
        dx_com -m <model.onnx> -c <config.json> -o <out_dir> [--shrink]
    Use the produced .dxnn with DX-RT or sample apps
    (Optional) Validate reproducibility with --shrink and fixed configurations

