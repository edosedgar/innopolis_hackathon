# Disc-type glass insulators absence detection

Team members: Edgar Kaziakhmedov, Denis Koposov  
Team name: NeuroEye

## Preface

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

Download [the model weights]((https://drive.google.com/file/d/1gweLmrbDAfyAiRBXGQC2RS2wiYdlCJ3f/view?usp=sharing)) from Google Drive and move the unzipped folder **models** to the same directory as your source code.

## Solution

We trained YoloV8x and did this and that (mention tile approach if it works out)

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