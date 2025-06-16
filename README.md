
# Image Denoising 😃🌟 

With the goal of recovering high-quality image content from its degraded version, image restoration enjoys numerous applications, such as in photography, security, medical imaging, and remote sensing. In this project I have implemented an model named MirNet for low-light image enhancement. It is a fully-convolutional architecture that learns an enriched set of features that combines contextual information from multiple scales, while simultaneously preserving the high-resolution spatial details

## Table of Contents 💡
- **Introduction**
- **PSNR value Achieved**
- **Pipeline**
- **How to install?**
- **More Knowledge**
## Introduction 🍁
This repository implements the MIRNet model for enhancing low-light images. MIRNet, which stands for Multi-Scale Residual Attention Network, is a powerful neural network designed to improve the visibility and quality of images captured in poor lighting conditions. This implementation includes training on the LOL dataset and testing on a custom test set to demonstrate the model's effectiveness.
## PSNR value
Peak Signal-to-Noise Ratio (PSNR) is a metric used to evaluate the quality of reconstructed or restored images, comparing them to the original images. It is widely used in image processing tasks such as image compression, denoising, and restoration. PSNR measures the ratio between the maximum possible power of a signal (image data) and the power of the corrupting noise that affects the fidelity of its representation. Higher PSNR values indicate better quality of the restored image, meaning it is closer to the original.

I am very glad to mention that in my MirNet Model Implementation i achieved PSNR value of:
### 1)At training data: 67.1063
### 2)At validation data: 67.2350
For further knowledge you can view the graph plotted for psnr vs epochs at
: https://drive.google.com/file/d/1QJcQBRKx0XBfAuozyn60RmsiKcFH5Qff/view?usp=sharing
## Pipeline ❄️
### 1. Data Preparation
- **Training Data**: Images are sourced from the LOL dataset, specifically the `our485` directory.
- **Testing Data**: Images are sourced from the `data/test` directory.
- **Preprocessing**: Images are read and normalized to a [0, 1] range, followed by random cropping to a fixed size of 128x128 pixels.

### 2. Model Architecture

#### 2.1 Input Layer
- The model accepts images of variable sizes with 3 channels (RGB).

#### 2.2 Recursive Residual Groups (RRG)
- Each RRG contains multiple Multi-Scale Residual Blocks (MRB).
- Each MRB integrates features from three different scales using down-sampling and up-sampling modules.

#### 2.3 Dual Attention Unit (DAU)
- Each MRB contains DAUs that incorporate both spatial and channel attention mechanisms to emphasize important features.

#### 2.4 Selective Kernel Feature Fusion (SKFF)
- SKFF modules are used to combine multi-scale features effectively by learning to weight features from different scales dynamically.

#### 2.5 Multi-Scale Residual Block (MRB)
- MRBs apply DAUs and SKFFs to learn robust features at multiple scales and improve feature representation.

### 3. Training

#### 3.1 Loss Function
- **Charbonnier Loss**: A smooth approximation of L1 loss, used for its robustness to outliers.

#### 3.2 Metrics
- **Peak Signal-to-Noise Ratio (PSNR)**: Used to measure the quality of the reconstructed image.

#### 3.3 Optimizer
- **Adam Optimizer**: Used for training with a learning rate of 1e-4.

#### 3.4 Training Process
- The model is trained for 50 epochs with a learning rate reduction strategy based on validation PSNR.

### 4. Inference
- The trained model is used to enhance low-light images.
- The inference pipeline includes reading the image, normalizing it, passing it through the model, and post-processing the output to generate the final enhanced image.

### 5. Evaluation
- Enhanced images are compared with original low-light images and images processed using autocontrast for qualitative assessment.

## Results
- The effectiveness of the MIRNet model is demonstrated through qualitative comparisons, showcasing significant improvements in image visibility and quality.

## How to Install my Repository and work with it on your system ?
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kanika-04/Low-Light-Image-Enhancement
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Low-Light-Image-Enhancement/
    next line:
    cd ./Project/
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

    ```
## Training

To train the model, run:
```bash
python main.py
```

## How to Install my Repository and work with it on your system ?
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kanika-04/Low-Light-Image-Enhancement
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Low-Light-Image-Enhancement/
    next line:
    cd ./Project/
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

    ```
## Training

To train the model, run:
```bash
python main.py
```
## More Knowledge
For More Knowledge:
U can view this report created by me :
https://drive.google.com/file/d/1QJcQBRKx0XBfAuozyn60RmsiKcFH5Qff/view?usp=sharing

Or you can view the research paper at:
https://www.ijfmr.com/papers/2023/3/4001.pdf
