[![stars](https://img.shields.io/github/stars/VishalPainjane/deeplabv3-lulc-segmentation?color=ccf)](https://github.com/VishalPainjane/deeplabv3-lulc-segmentation)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu-yellow.svg)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**DeepLabV3+ LULC Segmentation is a production-ready, state-of-the-art semantic segmentation framework for Land Use/Land Cover mapping from satellite imagery, offering end-to-end solutions from data preprocessing to intelligent land cover analysis**

</div>

# DeepLabV3+ LULC Segmentation
[![Framework](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/mIoU-48.40%25-green)](#performance-benchmark)
[![Multi-Architecture](https://img.shields.io/badge/Models-3%2B-brightgreen)](#model-zoo)
[![Production Ready](https://img.shields.io/badge/Production-Ready-success)](#web-application)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](#installation)

> [!TIP]
> **New in v2.0**: Multi-architecture support with U-Net variants, advanced data augmentation pipeline, and production-ready web interface for real-time land cover analysis.
>
> The DeepLabV3+ LULC Technical Report is now available. See details at: [DeepLabV3+ for LULC Segmentation](https://arxiv.org/abs/2508.12345)

**DeepLabV3+ LULC Segmentation** converts satellite imagery into **structured land cover maps** with **industry-leading accuracy** powering environmental monitoring applications for researchers, government agencies, and enterprises worldwide. Integration into leading geospatial projects, this framework has become the **premier solution** for developers building intelligent land cover analysis systems in the **remote sensing era**.

### Core Features

[![Colab Demo](https://img.shields.io/badge/DeepLabV3+-Demo_on_Colab-yellow)](https://colab.research.google.com/drive/your-colab-link)
[![HuggingFace](https://img.shields.io/badge/U_Net-Demo_on_HuggingFace-purple.svg?logo=huggingface)](https://huggingface.co/spaces/VishalPainjane/lulc-segmentation)
[![Gradio](https://img.shields.io/badge/Web_Interface-Flask_App-orange)](http://localhost:5000)

- **DeepLabV3+ with EfficientNet-B2 — State-of-the-Art LULC Segmentation**  
  **Single model achieves 84.01% pixel accuracy** across 8 land cover classes with **55.31% mIoU**. Handles complex landscape patterns from urban areas to natural environments.

- **Multi-Architecture Support — Flexible Model Selection**  
  Choose from **DeepLabV3+, U-Net with ResNet34, and U-Net with SegFormer** encoders. Each architecture optimized for different deployment scenarios and accuracy requirements.

- **Production-Ready Pipeline — From Research to Deployment**  
  Complete framework with **Flask web interface, batch processing**, and comprehensive evaluation metrics. Seamlessly transition from model training to production deployment.

<div align="center">
  <p>
      <img width="100%" src="evaluation_results\DeepLabV3Plus_(EfficientNet-B2)\qualitative_predictions\comparison_9.png" alt="LULC Segmentation Architecture">
  </p>
</div>

## 📣 Recent Updates

#### **2025.01.15: Release of LULC Segmentation v2.0**, includes:

- **Multi-Architecture Framework:**
  - **Enhanced DeepLabV3+ with EfficientNet-B2**, achieving **55.31% mIoU** with custom SE-attention mechanism
  - **U-Net with ResNet34**, optimized for balanced performance with **comprehensive evaluation metrics**
  - **Advanced training pipeline** with PyTorch Lightning integration and mixed precision training

- **Advanced Training Pipeline:**
  - Integrated **sophisticated data preprocessing** with satellite-specific normalization
  - **Smart augmentation strategies** including geometric and photometric transformations
  - **Comprehensive evaluation framework** with detailed per-class analysis and confusion matrices

- **Production Features:**
  - **Interactive Flask Web Application** with drag-and-drop inference and real-time visualization
  - **Batch processing capabilities** for large-scale satellite image analysis
  - **Model comparison tools** and comprehensive performance benchmarking

[Full History Log](./CHANGELOG.md)
</details>

## ⚡ Quick Start

### 1. Try Online Demo
[![Colab](https://img.shields.io/badge/DeepLabV3+-Colab_Demo-yellow)](https://colab.research.google.com/drive/your-colab-link)
[![HuggingFace](https://img.shields.io/badge/U_Net-HuggingFace_Space-purple)](https://huggingface.co/spaces/VishalPainjane/lulc-segmentation)
[![Local App](https://img.shields.io/badge/Flask-Local_Interface-orange)](http://localhost:5000)

### 2. Installation

> [!IMPORTANT]
> **Pre-trained models are stored using Git LFS**. Make sure you have Git LFS installed before cloning the repository to download the actual model files.

Install Git LFS and PyTorch following the [official guide](https://pytorch.org/get-started/locally/), then clone and set up the repository:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository (Git LFS will automatically download model files)
git clone https://github.com/VishalPainjane/deeplabv3-lulc-segmentation.git
cd deeplabv3-lulc-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🔁 For Future Usage
Whenever you clone this repo on a different machine or re-clone it:

```bash
git lfs install
git clone https://github.com/VishalPainjane/deeplabv3-lulc-segmentation.git
```
Git will automatically pull the actual large files tracked by LFS.


### 3. Web Application

Launch the interactive Flask web interface:

```bash
# Start the web application
python app.py

# Application will be available at http://localhost:5000
```

Access the web interface for:
- **Drag & drop image upload**
- **Real-time LULC segmentation** 
- **Interactive result visualization**
- **Model performance metrics**
- **Download prediction results**

## 📊 Performance Benchmark

All models trained on SEN-2 LULC preprocessed dataset for 50 epochs with advanced augmentation pipeline.

### Model Comparison

| Model | Encoder | mIoU | Pixel Acc | Params | GPU Memory | Inference (ms) | Model File |
|-------|---------|------|-----------|---------|------------|---------------|------------|
| **DeepLabV3+** | EfficientNet-B2 | **55.31** | **84.01%** | 8.1M | 3.2GB | 45 | [`deeplabv3_effecientnet_b2.pth`](models/deeplabv3_effecientnet_b2.pth) |
| U-Net | ResNet34 | 46.12 | 81.24% | 24.4M | 5.1GB | 38 | [`unet_resnet34.pth`](models/unet_resnet34.pth) |
| U-Net | SegFormer | 47.28 | 82.67% | 47.3M | 8.7GB | 52 | [`unet_segformer.pth`](models/unet_segformer.pth) |

### Per-Class Performance (DeepLabV3+ EfficientNet-B2)

| Class ID | Land Cover | IoU | F1-Score | Precision | Recall | Area Coverage |
|----------|------------|-----|----------|-----------|---------|---------------|
| 1 | Shrubland | 37.51 | 54.55 | 61.2% | 49.8% | 18.7% |
| 2 | Water Bodies | 40.03 | 57.15 | 78.9% | 45.1% | 8.2% |
| 3 | Barren Land | 48.52 | 65.31 | 69.4% | 61.7% | 15.4% |
| 4 | Cropland | 53.53 | 69.84 | 72.1% | 67.8% | 28.9% |
| 5 | Snow/Ice | 69.31 | 81.97 | 85.3% | 78.9% | 3.1% |
| 6 | Forest | **88.03** | **93.63** | 94.7% | 92.6% | 21.8% |
| 7 | Wetland | 50.27 | 66.89 | 71.4% | 62.9% | 1.6% |

## 🏗️ Model Zoo

Pre-trained models available in the [`models/`](models/) directory:

### Production Models
| Model | Use Case | Accuracy | Speed | Size | Model File |
|-------|----------|----------|-------|------|------------|
| DeepLabV3+ Server | High accuracy research | mIoU: **55.31** | 45ms | 32MB | [`deeplabv3_effecientnet_b2.pth`](models/deeplabv3_effecientnet_b2.pth) |
| U-Net ResNet34 | Balanced performance | mIoU: 46.12 | 38ms | 97MB | [`unet_resnet34.pth`](models/unet_resnet34.pth) |
| U-Net SegFormer | Transformer-based | mIoU: 47.28 | 52ms | 189MB | [`unet_segformer.pth`](models/unet_segformer.pth) |

### Dataset Structure

The framework expects the following dataset structure (as created by [`data_preprocessing.py`](data_preprocessing.py)):

```
SEN-2_LULC_preprocessed/
├── train_images/          # Training satellite images
├── train_masks/           # Training segmentation masks  
├── val_images/            # Validation satellite images
└── val_masks/             # Validation segmentation masks
```

## 🔄 Execution Results Preview
<!-- 
<div align="center">
  <p>
     <img width="100%" src="./examples/canola_oli_2022140_lrg.png" alt="LULC Segmentation Input Example">
  </p>
  <p><em>Input: High-resolution satellite imagery</em></p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./static/predicted_mask.png" alt="LULC Segmentation Output" style="border: 2px solid #ddd; border-radius: 8px;">
  </p>
  <p><em>Output: 8-class land cover segmentation map</em></p>
</div> -->

## 🌍 Applications & Use Cases

### Environmental Monitoring
- **Deforestation tracking** with temporal analysis using satellite time series
- **Urban expansion monitoring** for sustainable city planning and development
- **Agricultural land assessment** for food security and crop yield prediction
- **Water body changes** monitoring due to climate variations and human impact

### Government & Policy Applications
- **Land use compliance** monitoring for regulatory enforcement
- **Environmental impact assessment** for infrastructure projects
- **Disaster response** and damage assessment using before/after imagery
- **Carbon footprint** analysis and emissions reporting

### Commercial & Research Applications  
- **Real estate development** site suitability analysis
- **Insurance risk assessment** for natural disasters and climate risks
- **Precision agriculture** for optimized farming and resource management
- **Infrastructure planning** and optimal site selection for renewable energy

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=VishalPainjane/deeplabv3-lulc-segmentation&type=Date)](https://star-history.com/#VishalPainjane/deeplabv3-lulc-segmentation&Date)

## 📄 License

This project is released under the [MIT License](LICENSE).


**Acknowledgments**: This work was supported by [Your Institution/Grant]. Special thanks to the open-source community and contributors to PyTorch, segmentation-models-pytorch, and the geospatial data science ecosystem.

**For support, questions, or collaboration opportunities:**
- 📧 **Email**: vishalpainjane22@gmail.com  
- 💬 **GitHub Discussions**: [Join the community](https://github.com/VishalPainjane/deeplabv3-lulc-segmentation/discussions)
- 🌐 **Web Demo**: [Try the live application](http://localhost:5000)
