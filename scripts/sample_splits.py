import pandas as pd
from glob import glob
import cv2
import ast
import sys, os
import numpy as np
import shutil
from tqdm import tqdm

##################### CONFIG

DST_DS = 'complete_ds/v2'
TRN = 0.8
rs = np.random.RandomState(seed=1996)

sources = [
    'clean_ds/broken_glass_insulator640',
    'clean_ds/defect_ninja640',
    'clean_ds/imagestocks640',
    'clean_ds/innopolis640',
    'clean_ds/su110kv_broken640',
    'clean_ds/dolgoprudny_640',
    'clean_ds/web_crawl640',
    'clean_ds/mitino_live640'
]

#####################

file_pool = []
for source in sources:
    file_pool.extend(glob(source + '/images/*'))

file_pool = np.array(file_pool)
rs.shuffle(file_pool)
train_files = file_pool[0:int(TRN*len(file_pool))]
valid_files = file_pool[int(TRN*len(file_pool)):len(file_pool)]

# copy train files
os.makedirs(f'{DST_DS}/train/images', exist_ok=True)
os.makedirs(f'{DST_DS}/train/labels', exist_ok=True)
for train_file in train_files:
    filename = train_file.split('/')[-1][:-4]
    shutil.copy(train_file, f'{DST_DS}/train/images/{filename}.jpg')
    # dont copy for bg images
    if os.path.isfile('/'.join(train_file.split('/')[:-2]) + '/labels/' + filename + '.txt'):
        shutil.copy('/'.join(train_file.split('/')[:-2]) + '/labels/' + filename + '.txt',\
                    f'{DST_DS}/train/labels/{filename}.txt')

# copy valid files
os.makedirs(f'{DST_DS}/valid/images', exist_ok=True)
os.makedirs(f'{DST_DS}/valid/labels', exist_ok=True)
for valid_file in valid_files:
    filename = valid_file.split('/')[-1][:-4]
    shutil.copy(valid_file, f'{DST_DS}/valid/images/{filename}.jpg')
    # dont copy for bg images
    if os.path.isfile('/'.join(valid_file.split('/')[:-2]) + '/labels/' + filename + '.txt'):
        shutil.copy('/'.join(valid_file.split('/')[:-2]) + '/labels/' + filename + '.txt',\
                    f'{DST_DS}/valid/labels/{filename}.txt')