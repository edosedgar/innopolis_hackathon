# Disc-type glass insulators absence detection

Team name: NeuroEye  
Team members: Edgar Kaziakhmedov, Denis Koposov

## Introduction

The reliability of electrical networks depends on the presence and condition of insulators. The objective is to create an algorithm capable of automatically detecting cases where the glass insulator is missing in the strings of disc-type insulators based on RGB images.

## Installation

Install virtuenv package for python:
```sh
pip3 install virtualenv
```

Create new virtual environment and activate it:
```sh
python3 -m venv ~/kaggle_env
. ~/kaggle_env/bin/activate
```

Install required python packages:
```sh
pip3 install -r requirements.txt
```

## Model weights

Download [the model weights](https://drive.google.com/file/d/1gweLmrbDAfyAiRBXGQC2RS2wiYdlCJ3f/view?usp=sharing) from Google Drive and move the unzipped folder **models** to the same directory as your source code.

## Solution

In order to solve the problem, we use off-the-shelf YOLOv8x architecture, initially pre-trained and subsequently fine-tuned on our specific dataset, which consists of a single class. Network is trained on images size of 640x640 which are subjected to a range of augmentations to enhance model robustness.  
Throughout the training process,we configure a batch size of 16, utilizing gradient accumulation to effectively achieve a batch size equivalent to 64. For a comprehensive list of hyper-parameters employed in the training, please refer to [config file](configs/train/yolov8x_adamw_best.yaml).

## Evaluation

show how to run

## Data collection

The most important part is data collection. We gathered data from many sources (list them) and did this and that.
- [broken glass insulator](https://universe.roboflow.com/deep-learning-wpmkc/broken-glass-insulator) - 49 images
- [cach-dien-thuy](https://universe.roboflow.com/osu/cach-dien-thuy) + [su110kv_broken-sgwz3](https://universe.roboflow.com/osu/su110kv_broken-sgwz3) - 247 images
- [insulator-defect-detection](https://datasetninja.com/insulator-defect-detection#download) - 23 images
- images from shutterstock - 50 images
- photos of Moscow power lines - 122 + 233 = 355 images
- searching with Yandex/Google - 23 images
show some pics