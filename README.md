# Building Extraction using YOLO based Instance Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i_sEcacgnVJo56Z0CMe6flikIYKCPz4S?usp=sharing)

#### By [Yi Jie WONG](https://github.com/yjwong1999) & [Yin-Loon Khor](https://www.linkedin.com/in/yinloonkhor/) et al

This code is part of our solution for [2024 IEEE BigData Cup: Building Extraction Generalization Challenge (IEEE BEGC2024)](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview). Specifically, this repository provides the code to extract additional building footprint data from the Microsoft Building Footprint (BF) dataset for Redmond, Washington, and Las Vegas, Nevada. We use the extracted dataset to train our YOLOv8-based instance segmentation model, along with the training set provided by the IEEE BEGC2024 dataset. Results show that YOLOv8 trained on BEGC2024 with the additional dataset achieves a significant F1-score improvement compared to training on the BEGC2024 training set alone. Our approach ranked 1st globally in the IEEE Big Data Cup 2024 - BEGC2024 challenge! 🏅🎉🥳

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

### Training with Different Instance Segmentation Model
<table border="1" cellpadding="10" cellspacing="0">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Pretrained Weights</th>
      <th rowspan="2">Batch Size</th>
      <th rowspan="2">Params (M)</th>
      <th rowspan="2">FLOPs (G)</th>
      <th colspan="2">Public F1-Score</th>
    </tr>
    <tr>
      <th>Conf = 0.50</th>
      <th>Conf = 0.20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>YOLOv8n-seg</td>
      <td rowspan="4">DOTAv1 Aerial Detection</td>
      <td>16</td>
      <td>3.4</td>
      <td>12.6</td>
      <td>0.510</td>
      <td>0.645</td>
    </tr>
    <tr>
      <td>YOLOv8s-seg</td>
      <td>16</td>
      <td>11.8</td>
      <td>42.6</td>
      <td>0.535</td>
      <td>0.654</td>
    </tr>
    <tr>
      <td>YOLOv8m-seg</td>
      <td>16</td>
      <td>27.3</td>
      <td>110.2</td>
      <td>0.592</td>
      <td>0.649</td>
    </tr>
    <tr>
      <td>YOLOv8x-seg</td>      
      <td>8</td>
      <td>71.8</td>
      <td>344.1</td>
      <td>0.579</td>
      <td>0.627</td>
    </tr>
    <tr>
      <td>YOLOv9c-seg</td>
      <td>COCO Segmentation</td>
      <td>4</td>
      <td>27.9</td>
      <td>159.4</td>
      <td>0.476</td>
      <td>0.577</td>
    </tr>
    <tr>
      <td>Mask R-CNN (MPViT-Tiny)</td>
      <td>COCO Segmentation</td>
      <td>4</td>
      <td>17</td>
      <td>196.0</td>
      <td>-</td>
      <td>0.596</td>
    </tr>
    <tr>
      <td>EfficientNet-b0-YOLO-seg</td>
      <td>ImageNet</td>
      <td>4</td>
      <td>6.4</td>
      <td>12.5</td>
      <td>-</td>
      <td>0.560</td>
    </tr>
  </tbody>
</table>


### Training with Different Dataset
<table border="1">
  <tr>
    <th rowspan=2>Solution</th>
    <th rowspan=2>FLOPS (G)</th>
    <th colspan="2">F1-Score</th>
  </tr>
  <tr>
    <td>Public</td>
    <td>Private</td>
  </tr>
  <tr>
    <td>YOLOv8m-seq + BEGC 2024</td>
    <td rowspan=4>110.2</td>
    <td>0.64926</td>
    <td>0.66531</td>
  </tr>
  <tr>
    <td>YOLOv8m-seq + BEGC 2024 + Redmond Dataset</td>
    <td>0.65951</td>
    <td>0.67133</td>
  </tr>
  <tr>
    <td>YOLOv8m-seq + BEGC 2024 + Las Vegas Dataset</td>
    <td>0.68627</td>
    <td>0.70326</td>
  </tr>
  <tr>
    <td>YOLOv8m-seq + BEGC 2024 + Diffusion Augmentation</td>
    <td>0.67189</td>
    <td>0.68096</td>
  </tr>
  <tr>
    <td>2nd place (RTMDet-x + Alabama Buildings Segmentation Dataset)</td>
    <td>141.7</td>
    <td>0.6813</td>
    <td>0.68453</td>
  </tr>
  <tr>
    <td>3rd Place (Custom Mask-RCNN + No extra Dataset)</td>
    <td>124.1</td>
    <td>0.59314</td>
    <td>0.60649</td>
  </tr>
</table>

Refer [our segmentation-guided diffusion model](https://github.com/yjwong1999/RSGuidedDiffusion) to see how we implement our diffusion augmentation pipeline.

### Inference with Different NMS IoU Threshold 
<table border="1" cellpadding="10" cellspacing="0">
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="6">Private F1 Score</br>(using different NMS IoU Threshold)</th>
    </tr>
    <tr>
      <th>0.70</th>
      <th>0.75</th>
      <th>0.80</th>
      <th>0.85</th>
      <th>0.90</th>
      <th>0.95</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BEGC2024 + Redmond Dataset</td>
      <td>0.672</td>
      <td>0.677</td>
      <td>-</td>
      <td>-</td>
      <td>0.748</td>
      <td>0.866</td>
    </tr>
    <tr>
      <td>BEGC2024 + Las Vegas Dataset</td>
      <td>0.703</td>
      <td>0.693</td>
      <td>0.686</td>
      <td>0.721</td>
      <td>0.766</td>
      <td>0.897</td>
    </tr>
    <tr>
      <td>BEGC2024 + Diffusion Augmentation</td>
      <td>0.681</td>
      <td>-</td>
      <td>0.694</td>
      <td>0.711</td>
      <td>0.751</td>
      <td>0.887</td>
    </tr>
  </tbody>
</table>


## Acknowledgement
We thank the following works for the inspiration of our repo!
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Ultralytic YOLO [code](https://github.com/ultralytics/ultralytics)
3. MPViT-based Mask RCNN [code](https://github.com/youngwanLEE/MPViT)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)
