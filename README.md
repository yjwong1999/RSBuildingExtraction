# Building Extraction using YOLO based Instance Segmentation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1i_sEcacgnVJo56Z0CMe6flikIYKCPz4S?usp=sharing)

#### By [Yi Jie WONG](https://github.com/yjwong1999) & [Yin Look Khor](https://www.linkedin.com/in/yinloonkhor/) et al

This code is part of our solution for 2024 IEEE BigData Cup: Building Extraction Generalization Challenge (BEGC).

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

## Acknowledgement
We thank the following works for the inspiration of our repo!
1. 2024 IEEE BigData Cup: Building Extraction Generalization Challenge [link](https://www.kaggle.com/competitions/building-extraction-generalization-2024/overview)
2. Ultralytic YOLO [code](https://github.com/ultralytics/ultralytics)
3. MPViT-based Mask RCNN [code](https://github.com/youngwanLEE/MPViT)
4. COCO2YOLO format [original code](https://github.com/tw-yshuang/coco2yolo), [modified code](https://github.com/yjwong1999/coco2yolo)
