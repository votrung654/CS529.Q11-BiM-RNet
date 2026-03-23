<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS529.Q11 - RESEARCH ISSUES AND APPLICATIONS IN COMPUTER SCIENCE</b></h1>

<p align="center">
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square" alt="PyTorch"></a>
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat-square" alt="Python"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Research%20Only-blue.svg?style=flat-square" alt="License"></a>

</p>

## TABLE OF CONTENTS
* [Course Introduction](#course-introduction)
* [Instructor](#instructor)
* [Students](#students)
* [Project](#project)
* [System Requirements](#system-requirements)
* [Environment Setup](#environment-setup)
* [Important Configuration Notes](#important-configuration-notes)
* [Data](#data)
* [Pre-trained Models](#pre-trained-models)
* [Evaluation](#evaluation)
* [Training](#training)
* [Demo](#demo)
  - [Command Line Demo](#command-line-demo)
  - [Web Demo](#web-demo)
* [Kaggle Notebooks](#kaggle-notebooks)
* [References](#references)

## COURSE INTRODUCTION
<a name="course-introduction"></a>
* **Course Name**: Research Issues and Applications in Computer Science
* **Course Code**: CS529
* **Class Code**: CS529.Q11
* **Start Date**: 09/08/2025
* **Academic Year**: 2025 - 2026

## INSTRUCTOR
<a name="instructor"></a>
* **MSc. Do Van Tien** - *tiendv@uit.edu.vn*

## STUDENTS
<a name="students"></a>
| Student ID | Full Name | Github | Email |
|:----------:|:-------------------:|:----------------------------------------------------:|:-----------------------:|
| 22521587   | Truong Phuc Truong  | [Truong99zvc](https://github.com/Truong99zvc/)      | 22521587@gm.uit.edu.vn  |
| 22521571   | Vo Dinh Trung       | [votrung654](https://github.com/votrung654/)         | 22521571@gm.uit.edu.vn  |

## PROJECT
<a name="project"></a>
**Project Name**: EVALUATION, REPRODUCTION, AND EXTENSION OF TWO-DIMENSIONAL MOTION FIELD-GUIDED VIDEO FRAME INTERPOLATION FOR COMPLEX MOTION SCENARIOS

**This project is the result of reproduction, evaluation, and extension based on the research:**
> **BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions**
> *Authors: Wonyong Seo, Jihyong Oh, Munchurl Kim*

This repository contains the implementation and evaluation source code of two models:
1.  **BiM-VFI (Reproduced)**: The original method from the paper, fully reproduced.
2.  **BiM-RNet (Proposed/Extension)**: The proposed improved architecture, increasing performance and optimizing feature extraction capabilities in complex motion scenarios.

**Student Group Contributions:**
1.  **Reproducibility:** Retrained the original BiM-VFI model from scratch on the Vimeo90K dataset to verify the results.
2.  **Proposed Extension (BiM-RNet):** Designed and implemented the **BiM-RNet** architecture, leveraging the power of ResNet Encoder.
3.  **Compatibility Fixes:** Modified the source code to run stably on modern environments (new Torch, new CUDA) and the Windows operating system.
4.  **Application Development:** Integrated a **Web Demo** module (Flask) for intuitive interaction.
5.  **Documentation:** Detailed and translated the user guide.

The core source code is based on the original repository of the authors' group KAIST-VICLab.

## SYSTEM REQUIREMENTS
<a name="system-requirements"></a>
To run the source code stably, the system must meet the following minimum requirements:

*   **Operating System**: Windows 10/11 (thoroughly tested) or Linux.
*   **GPU (Important)**:
    *   **NVIDIA GPU** with CUDA support is mandatory.
    *   **VRAM**: Minimum **4GB** (Tested stably on GTX 1650 at low/medium resolutions). Recommended **8GB+** for HD/FullHD video inference.
    *   **Note**: For training, especially BiM-RNet, using a powerful GPU server (like Kaggle P100/T4x2 or better) is recommended as 4GB VRAM is insufficient for a large batch size.
*   **CUDA Driver**: Requires the latest NVIDIA Driver compatible with CUDA 11.8 or higher.
*   **RAM**: Minimum 16GB (Video processing consumes a lot of RAM).

## ENVIRONMENT SETUP
<a name="environment-setup"></a>

### Prerequisites
Before setting up the environment, ensure **Conda** is installed on your system. You can download and install Conda from:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Recommended - lightweight)
- [Anaconda](https://www.anaconda.com/products/distribution) (Full version)

### Environment Setup and Library Installation

> **Important Note**: The library versions in this repository differ from the original BiM-VFI repository. To ensure reproducibility and stable execution on mainstream GPUs like **GTX 1650**, we use the latest PyTorch version supporting CUDA 13.0.

```bash
conda create -n bimvfi python=3.11
conda activate bimvfi
pip install basicsr-fixed Ipython torchsummary wandb moviepy pyyaml imageio packaging tqdm opencv-python tensorboardx ptflops pyiqa lpips stlpips_pytorch dists_pytorch torch torchvision --index-url https://download.pytorch.org/whl/cu130
conda install cupy -c conda-forge
```

### Additional Libraries for Web Demo
To run the web demo, install the following additional libraries:

```bash
pip install flask werkzeug pillow scikit-image
```

**Note**: `opencv-python` and `torch` are already included in the main environment setup above. Ensure they are fully installed.

## IMPORTANT CONFIGURATION NOTES
<a name="important-configuration-notes"></a>

### Environment Variable KMP_DUPLICATE_LIB_OK
In `main.py`, we added a configuration to resolve OpenMP conflicts:
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
This environment variable handles the "OMP: Error #15: Initializing libiomp5md.dll..." error commonly encountered on Windows when using NumPy, PyTorch, and OpenCV together.

### Absolute Path Configuration

After cloning the repository, you **must** modify the dataset and model paths in the `cfgs/*.yaml` configuration files. Default relative paths often do not work correctly on Windows and should be changed to **absolute paths**.

#### For Training (`cfgs/bim_rnet.yaml` or `cfgs/bim_vfi.yaml`):
```yaml
# Before (will NOT work)
root_path: ../data/vimeo_triplet

# After (example - adjust to actual path)
root_path: D:/Github/BiM-RNet/data/vimeo_triplet
```

#### For Evaluation (`cfgs/bim_vfi_benchmark.yaml`):
```yaml
# Before
resume: ./save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth

# After (example)
resume: D:/Github/BiM-RNet/save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth
root_path: D:/Github/BiM-RNet/data/vimeo_triplet
```

## DATA
<a name="data"></a>
### Download
The project uses the Vimeo90K dataset for training and testing:
> - [Vimeo90K](https://cove.thecvf.com/datasets/875)

### Preparation
After downloading, extract it and place it in the `data` directory according to the project structure. We use the Vimeo 90K-Triplet dataset.

## PRE-TRAINED MODELS
<a name="pre-trained-models"></a>

This repository includes checkpoints:

### 1. Original Pre-trained Model (BiM-VFI)
- **Path**: `pretrained/bim_vfi.pth`
- **Description**: Original pre-trained model from the paper for comparison and baseline.

### 2. BiM-RNet (Proposed Model)
- **Path**: `save/bim_rnet_train_new__400_epochs_NEW/checkpoints/model_best.pth`
- **Description**: Our proposed improved model, trained on Vimeo90K.

## EVALUATION
<a name="evaluation"></a>
Evaluation is configured through the `cfgs/bim_vfi_benchmark.yaml` file.

*   **Config**: Adjust `name` (dataset name) and `root_path` (absolute data path) in the configuration file.
*   **Model**: Ensure the `resume` parameter points to the correct checkpoint to evaluate (original BiM-VFI or BiM-RNet).

Run the command:
```bash
python main.py --cfg cfgs/bim_vfi_benchmark.yaml
```

## TRAINING
<a name="training"></a>
The project supports training both the original BiM-VFI model and the proposed BiM-RNet model.

### 1. Train BiM-RNet (Improved Model)
Use configuration `cfgs/bim_rnet.yaml`.

**1 GPU Training:**
```bash
python main.py --cfg cfgs/bim_rnet.yaml
```

**Multi-GPU Training (Example 4 GPUs: 0, 1, 2, 3):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_rnet.yaml
```

### 2. Train BiM-VFI (Original Model / Reproduce)
Use configuration `cfgs/bim_vfi.yaml`.

**1 GPU Training:**
```bash
python main.py --cfg cfgs/bim_vfi.yaml
```

**Multi-GPU Training:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main.py --cfg cfgs/bim_vfi.yaml
```

To use **WandB** for tracking, fill in the details in `wandb.yaml` and add the `-w` flag.

## DEMO
<a name="demo"></a>
### Command Line Demo
<a name="command-line-demo"></a>
Custom videos in multiple image or video formats can be interpolated as follows.

First, set up the demo root directory like this:
  - video1.mp4 
  - video2.mp4
  - video3
    - img0.png
    - img1.png
    - ...

Then, replace `root_path` in `cfgs/bim_vfi_demo.yaml` with your desired data path, and run:
```bash
python main.py --cfg cfgs/bim_vfi_demo.yaml
```

### Web Demo
<a name="web-demo"></a>
The project includes a web-based demo interface for easy video frame interpolation, supporting intuitive interaction.

1.  **Navigate to the web_demo directory**:
    ```bash
    cd web_demo
    ```
2.  **Run the Flask application**:
    ```bash
    python app.py
    ```
3.  **Access the web interface**:
    Open a browser at `http://localhost:5000`

**Features:**
*   **Image Pair Interpolation**: Upload 2 images to generate the intermediate frame.
*   **Video Interpolation**: Increase frame rate for uploaded videos.
*   **Frame Sequence Interpolation**: Smoothly process sequences of multiple images.
*   **Model Selection**: Option to run BiM-VFI (Original) or BiM-RNet (Improved).
*   **Statistics**: View real processing time.

## KAGGLE NOTEBOOKS
<a name="kaggle-notebooks"></a>

For users **with incompatible GPUs, no GPUs**, or those who want to train/evaluate in the cloud, we provide sample Kaggle notebooks for each model:

> **1. BiM-VFI Notebook (Reproduce)**: [https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440](https://www.kaggle.com/code/truong9/bim-vfi?scriptVersionId=289083440)
>
> **2. BiM-RNet Notebook (Proposed)**: [https://www.kaggle.com/code/gtekx9/bim-rnet](https://www.kaggle.com/code/gtekx9/bim-rnet)

**Note**: The notebooks are pre-configured with the environment and dataset (Vimeo90K) on Kaggle, users just need to copy and run (using P100 or T4x2 GPU).

## REFERENCES
<a name="references"></a>
This project uses and extends source code from the [KAIST-VICLab/BiM-VFI](https://github.com/KAIST-VICLab/BiM-VFI) repository. We sincerely thank the authors.

The source code is referenced, refined, and developed for scientific research and educational purposes.

If using this source code for research, please cite the original paper:

```bibtex
@inproceedings{Seo_2025_CVPR,
  title={BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions},
  author={Seo, Wonyong and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={7244--7253},
  year={2025}
}
```

