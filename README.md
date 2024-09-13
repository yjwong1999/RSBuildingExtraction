# Building Extraction using YOLO based Instance Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i_sEcacgnVJo56Z0CMe6flikIYKCPz4S?usp=sharing)

#### By [Yi Jie WONG](https://github.com/yjwong1999) & [Yin Look Khor](https://www.linkedin.com/in/yinloonkhor/) et al

This code is part of our solution for [2024 IEEE BigData Cup: Building Extraction Generalization Challenge (IEEE BEGC2024)](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview). Specifically, this repository provides the code to extract additional building footprint data from the Microsoft Building Footprint (BF) dataset for Redmond, Washington, and Las Vegas, Nevada. We use the extracted dataset to train our YOLOv8-based instance segmentation model, along with the training set provided by the IEEE BEGC2024 dataset. Results show that YOLOv8 trained on BEGC2024 with the additional dataset achieves a significant F1-score improvement compared to training on the BEGC2024 training set alone.

## Instructions
Conda environment
```bash
conda create --name yolo python=3.10.12 -y
conda activate yolo
```

Clone this repo
```bash
# clone this repo
git clone https://github.com/yjwong1999/RSBuildingExtraction.git
cd RSBuildingExtraction
```

Install dependencies
```bash
# Please adjust the torch version accordingly depending on your OS
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# Install Jupyter Notebook
pip install jupyter notebook==7.1.0

# Remaining dependencies (for instance segmentation)
pip install ultralytics==8.1
pip install pycocotools
pip install requests==2.32.3
pip install click==8.1.7
pip install opendatasets==0.1.22
```

## Results
<table border="1">
    <tr>
        <th rowspan=2></th>
        <th colspan="2">F1 Score</th>
    </tr>
    <tr>
        <th>Public</th>
        <th>Private</th>
    </tr>
    <tr>
        <td>BEGC2024 dataset</td>
        <td>0.64926</td>
        <td>0.66331</td>
    </tr>
    <tr>
        <td>BEGC2024 dataset + Washington dataset</td>
        <td>0.65961</td>
        <td>0.67153</td>
    </tr>
    <tr>
        <td>BEGC2024 dataset + Las Vegas dataset</td>
        <td>0.68627</td>
        <td>0.70326</td>
    </tr>
    <tr>
        <td>BEGC2024 dataset + Diffusion Augmentation</td>
        <td>0.67189</td>
        <td>0.68096</td>
    </tr>
    <tr>
        <td>Second place (unknown model/dataset)</td>
        <td>0.6813</td>
        <td>0.68453</td>
    </tr>
    <tr>
        <td>Third place (unknown model/dataset)</td>
        <td>0.59314</td>
        <td>0.60649</td>
    </tr>
</table>

Refer [our segmentation-guided diffusion model](https://github.com/yjwong1999/RSGuidedDiffusion) to see how we implement our diffusion augmentation pipeline.

## Acknowledgement
We thank the following works for the inspiration of our repo!
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Ultralytic YOLO [code](https://github.com/ultralytics/ultralytics)
3. MPViT-based Mask RCNN [code](https://github.com/youngwanLEE/MPViT)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)
