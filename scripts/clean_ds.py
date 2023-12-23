import pandas as pd
from glob import glob
import cv2
import ast
import sys, os
import numpy as np
import shutil
from tqdm import tqdm

DS_NAME = 'su110kv_broken'

for subset in ['train', 'test', 'valid']:
    dst_ds = f'downloaded_datasets/{DS_NAME}'
    ds_name = f'downloaded_datasets/{DS_NAME}/{subset}/'

    labels = glob(f'{ds_name}/labels/*.txt')
    os.makedirs(f'{dst_ds}/images', exist_ok=True)
    os.makedirs(f'{dst_ds}/labels', exist_ok=True)

    for label in tqdm(labels):
        name = label.split('/')[-1][:-4]
        shutil.copy(f'{ds_name}/images/{name}.jpg', f'{dst_ds}/images/{name}.jpg')
        shutil.copy(f'{ds_name}/labels/{name}.txt', f'{dst_ds}/labels/{name}.txt')
