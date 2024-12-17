# MIRNet for Low-Light Image Enhancement

This repository implements and experiments with **MIRNet**, a deep learning architecture for enhancing low-light images. The project is inspired by the work presented in [Learning Enriched Features for Real Image Restoration and Enhancement](https://arxiv.org/abs/2003.06792) and was developed as part of the **EE610: Image Processing** course under the guidance of **Prof. Amit Sethi** at **IIT Bombay**.

---

## Table of Contents

1. [Introduction](#introduction)
2. [MIRNet Architecture](#mirnet-architecture)
3. [Experiments Conducted](#experiments-conducted)
4. [Results](#results)
5. [Dataset](#dataset)
6. [Installation](#installation)
7. [Usage](#usage)
8. [References](#references)

---

## Introduction

Low-light conditions degrade image quality, resulting in issues like noise, low contrast, and color distortion. Enhancing such images is a challenging problem in computer vision. This project uses **MIRNet**, a multi-scale convolutional neural network that:

- Captures spatial accuracy.
- Reduces noise and preserves details.
- Improves perceptual quality using hybrid loss functions.

The work focuses on restoring high-quality content from low-light images, particularly using the **LoL Dataset**.

---

## MIRNet Architecture

MIRNet introduces the following key components:
1. **Multi-Scale Residual Block (MRB)**: Captures multi-scale contextual information while maintaining high-resolution details.
2. **Dual Attention Unit (DAU)**: Combines channel attention and spatial attention for better feature refinement.

The network is trained using a **hybrid loss function** combining:
- **Charbonnier loss**: Robust loss for image restoration.

Optimizer: **Adam**  
Learning Rate Scheduler: **ReduceLROnPlateau**

---

## Experiments Conducted

### 1. Baseline Implementation
- Implemented MIRNet with original architecture and parameters.
- **Loss Function**: Charbonnier Loss.

### 2. Modified Architecture
- Experimented with **Dual Attention Units (DAU)** for refining features.
- Improved **SSIM** and **PSNR** metrics.

### 3. Alternative Loss Functions
- Added **SSIM Loss** alongside the Charbonnier loss to emphasize perceptual quality.\

### 4. Added regularization
- Experimented with the addition of **L1 and L2** regularization 

Each experiment can be found in:
- `mirnet.ipynb`: Baseline implementation.
- `mirnet_modified.ipynb`: Modified architecture.
- `mirnet_modified_SSIM_loss.ipynb`: Alternative loss functions.
- `mirnet_modified_regularisation.ipynb`: Added regularization.

---

### Sample Outputs

| **Input** | **Enhanced (Baseline)** | **Enhanced (Modified)** |
|-----------|-------------------------|--------------------------|
| ![Input](Test_images/input1.jpg) | ![Baseline](Results/baseline1.jpg) | ![Modified](Results/modified1.jpg) |
| ![Input](Test_images/input2.jpg) | ![Baseline](Results/baseline2.jpg) | ![Modified](Results/modified2.jpg) |

---

## Dataset

The **LoL Dataset** was used for training and evaluation. It contains paired low-light and well-exposed images:
- **Training**: 485 image pairs.
- **Testing**: 15 image pairs.

Dataset link: [LoL Dataset](https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6)

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TheShiningVampire/MIRNET_for_low_light_image_improvement.git
   cd MIRNET_for_low_light_image_improvement
   ```

2. **Set up the environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Usage

To run the experiments, execute the following Jupyter notebooks:

1. **Baseline Implementation**:
   ```bash
   jupyter notebook mirnet.ipynb
   ```

2. **Modified Architecture**:
   ```bash
   jupyter notebook mirnet_modified.ipynb
   ```

3. **SSIM Loss Implementation**:
   ```bash
   jupyter notebook mirnet_modified_SSIM_loss.ipynb
   ```

---

## References

1. [Learning Enriched Features for Real Image Restoration and Enhancement](https://arxiv.org/abs/2003.06792)
2. [Keras MIRNet Example](https://keras.io/examples/vision/mirnet/)
3. [LoL Dataset](https://drive.google.com/uc?id=1DdGIJ4PZPlF2ikl8mNM9V-PdVxVLbQi6)
