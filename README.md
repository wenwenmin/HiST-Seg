# HiST-Seg
## Introduction
Pixel-wise segmentation of cancerous tissue regions (CTRs) is important for pathological assessment and spatial molecular analysis. Existing methods fall into two categories: those that identify abnormal regions from histology alone and those that integrate histology with spatial transcriptomics (ST) data for spot-level CTR prediction. However, histology-only models lack molecular evidence, whereas spot-level methods are constrained by the coarse spatial resolution of ST and cannot recover precise CTR boundaries. We propose HiST-Seg, a local--regional dual-stream framework for pixel-wise CTR segmentation using H\&E images and ST. Specifically, the local stream extracts fine-grained morphology and boundary cues from high-resolution H\&E patches using a pretrained pathology encoder and multi-scale convolutional features. In parallel, the regional stream jointly encodes an enlarged co-centered H\&E field and a dense ST feature map constructed from spot-level transcriptomic features. An adaptive cross-resolution fusion mechanism integrates local pathological semantics with regional morphological and molecular information, followed by a region-guided decoder that reconstructs fine CTR boundaries. HiST-Seg can be trained on multiple source sections of the same cancer type and directly applied to unseen target sections without target-specific fine-tuning. Experiments on five ST datasets comprising 23 breast cancer and colorectal cancer tissue sections demonstrate that HiST-Seg achieves superior performance in CTR segmentation and cross-section generalization.

![Fig2_architecture.jpg](Fig2_V4_du.jpg)

## Requirements
All experiments were conducted on an NVIDIA RTX 3090 GPU. Before running HiST-Seg, you need to create a conda environment and install the required packages:
```shell
conda create -n HiST-Seg python==3.9.23
conda activate HiST-Seg
pip install -r requirements.txt
```

## Datasets
The ST human HER2-positive breast tumor datasets (STHBC) are available in: [https://github.com/almaan/her2st](https://github.com/almaan/her2st).

The 10X Visium Human Breast Cancer Breast Cancer: Ductal Carcinoma In Situ, Lobular carcinoma In Situ, Invasive Carcinoma dataset (ViHBC) are available in: [https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0](https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0).

The Xenium Human Breast Cancer dataset (XeHBC) can be found at: [https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).

The 10X Visium Human Breast Cancer: Ductal Carcinoma In Situ, Invasive Carcinoma dataset (DuCIS) are available in: [https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0](https://www.10xgenomics.com/datasets/human-breast-cancer-ductal-carcinoma-in-situ-invasive-carcinoma-ffpe-1-standard-1-3-0).

The 10X Visium Colorectal Cancer Visium datasets (CRC) are available in: [https://zenodo.org/records/7760264](https://zenodo.org/records/7760264).



## Pre-trained general-purpose foundation mode
Given the outstanding performance of large pre-trained general-purpose foundation models in clinical tasks, we utilize UNI as the backbone feature extractor specifically for one branch within the Local Stream. Before using HiST-Seg, you need to apply for permission to access the UNI model weights at:  [https://huggingface.co/mahmoodlab/UNI](https://huggingface.co/mahmoodlab/UNI).

For spatial transcriptomics, please refer to the scGPT-spatial installation guide：[https://github.com/bowang-lab/scGPT-spatial](https://github.com/bowang-lab/scGPT-spatial). Once the installation is complete, you can directly run the ''data_preprocess/scgpt_Extractor_XXX.py'' script to extract gene features.

## HiST-Seg pipeline
- Install the UNI module form [https://github.com/mahmoodlab/UNI](https://github.com/mahmoodlab/UNI).
- Install the scGPT-sptial from [https://huggingface.co/mahmoodlab/UNI](https://github.com/bowang-lab/scGPT-spatial).
- Run data_preprocess/crop_XXX.py
- Run data_preprocess/patching_XXX.py
- Run data_preprocess/scgpt_Extractor_XXX.py
- Run engine/train_XXX.py
- Run engine/inference_XXX.py
- Run downstream

