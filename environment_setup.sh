#!/bin/bash

set -e

ENV_NAME="simmlm"
PYTHON_VERSION="python=3.9.19"

echo ">>> Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME $PYTHON_VERSION

echo ">>> Activating conda environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ">>> Installing PyTorch 2.2.2 + CUDA 11.8 (pip, official command)"
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

echo ">>> Installing additional python packages (with fixed versions)"
pip install \
    numpy==1.26.4 \
    monai==1.3.0 \
    tqdm==4.65.0 \
    torch-summary==1.4.4 \
    SimpleITK==2.3.1 \
    pytorch-lightning==1.8.4 \
    scipy==1.13.1

echo ">>> Environment setup complete!"
