import pandas as pd
from glob import glob
import cv2
import ast
import sys, os
import numpy as np
from tqdm import tqdm

SRC_DS_labels = './downloaded_datasets/mitino_live/labels'
SRC_DS_images = './downloaded_datasets/mitino_live/images'
DST_DS = './downloaded_datasets/mitino_live640'
OFFSET = 10
TILE_SIZE = 960
FINAL_SIZE = 640

# SRC_DS_labels = 'downloaded_datasets/broken-glass-insulator.v9i.yolov8/labels'
# SRC_DS_images = 'downloaded_datasets/broken-glass-insulator.v9i.yolov8/images'
# DST_DS = 'clean_ds/broken_glass_insulator640'
# OFFSET = 50
# TILE_SIZE = 500
# FINAL_SIZE = 640

# SRC_DS_labels = 'innopolis-high-voltage-challenge'
# SRC_DS_images = 'innopolis-high-voltage-challenge'
# DST_DS = 'clean_ds/innopolis640'
# OFFSET = 300
# TILE_SIZE = 1000
# FINAL_SIZE = 640

## sample slice coordinates given some label position
## make sure coordinates are within image size
def sample_slice_position(rs, orig_x, orig_y, slice_width, slice_height, img_w, img_h, random_offset):
    orig_x = orig_x + int(np.round(rs.uniform(-random_offset, +random_offset)))
    orig_y = orig_y + int(np.round(rs.uniform(-random_offset, +random_offset)))

    x1, x2 = int(round(orig_x - slice_width/2)), int(round(orig_x + slice_width/2))
    y1, y2 = int(round(orig_y - slice_height/2)), int(round(orig_y + slice_height/2))

    if x1 < 0:
        x2 = x2 + abs(x1)
        x1 = 0
    if x2 >= img_w:
        x1 = x1 - abs(img_w - x2)
        x2 = img_w

    if y1 < 0:
        y2 = y2 + abs(y1)
        y1 = 0
    if y2 >= img_h:
        y1 = y1 - abs(img_h - y2)
        y2 = img_h
    return x1, x2, y1, y2

## transform label coordinates
def transform_labels(x1, x2, y1, y2, labels, img_w, img_h, is_visited):
    new_labels = []
    slice_w, slice_h = x2-x1, y2-y1

    for lbl_pos, label in enumerate(labels):
        box_x1, box_x2 = (label[0] - label[2]/2)*img_w, (label[0] + label[2]/2)*img_w
        box_y1, box_y2 = (label[1] - label[3]/2)*img_h, (label[1] + label[3]/2)*img_h

        # skip if label is outside selected window/slice
        if box_x1 >= x2 or box_x2 <= x1 or box_y1 >= y2 or box_y2 <= y1:
            continue

        new_box_x1, new_box_x2 = max(0, box_x1 - x1), min(slice_w, box_x2 - x1)
        new_box_y1, new_box_y2 = max(0, box_y1 - y1), min(slice_h, box_y2 - y1)

        # skip if new label is half less its original size along x or y
        if new_box_x2 - new_box_x1 < (box_x2 - box_x1)/2 or new_box_y2 - new_box_y1 < (box_y2 - box_y1)/2:
            continue

        is_visited[lbl_pos] = 1
        new_box_cx, new_box_cy = (new_box_x1 + new_box_x2)/2/slice_w, (new_box_y1 + new_box_y2)/2/slice_h
        new_box_w, new_box_h = (new_box_x2 - new_box_x1)/slice_w, (new_box_y2 - new_box_y1)/slice_h
        new_labels.append([new_box_cx, new_box_cy, new_box_w, new_box_h])

    return new_labels

os.makedirs(f'{DST_DS}/images', exist_ok=True)
os.makedirs(f'{DST_DS}/labels', exist_ok=True)

rs = np.random.RandomState(seed=1996)

src_labels = sorted(glob(f'{SRC_DS_labels}/*.txt'))
for file in tqdm(src_labels):
    filename = file.split("/")[-1][:-4]
    if os.path.isfile(f'{SRC_DS_images}/{filename}.JPG'):
        img = cv2.imread(f'{SRC_DS_images}/{filename}.JPG')
    elif os.path.isfile(f'{SRC_DS_images}/{filename}.jpg'):
        img = cv2.imread(f'{SRC_DS_images}/{filename}.jpg')
    elif os.path.isfile(f'{SRC_DS_images}/{filename}.jpeg'):
        img = cv2.imread(f'{SRC_DS_images}/{filename}.jpeg')
    elif os.path.isfile(f'{SRC_DS_images}/{filename}.png'):
        img = cv2.imread(f'{SRC_DS_images}/{filename}.png')
    else:
        print("Error reading file")
        sys.exit(1)

    img_h, img_w, _ = img.shape

    with open(file) as f:
        lines = [list(map(float, line.split(" ")[1:])) for line in f]
    if len(lines) == 0:
        continue

    is_visited = np.zeros(len(lines))
    for i, line in enumerate(lines):
        if is_visited[i] == 1:
            continue
        x_c, y_c, box_w, box_h = line[0]*img_w, line[1]*img_h, line[2]*img_w, line[3]*img_h

        cur_tile_size = min(TILE_SIZE, min(img_h, img_w))
        x1, x2, y1, y2 = sample_slice_position(rs, x_c, y_c, cur_tile_size, cur_tile_size,\
                                               img_w, img_h, random_offset=OFFSET)

        img_window = img[y1:y2,x1:x2]
        img_window = cv2.resize(img_window, (FINAL_SIZE, FINAL_SIZE), interpolation = cv2.INTER_AREA)
        cv2.imwrite(f'{DST_DS}/images/{filename}_{i}.JPG', img_window)

        with open(f'{DST_DS}/labels/{filename}_{i}.txt', 'w+') as f:
            new_labels = transform_labels(x1, x2, y1, y2, lines, img_w, img_h, is_visited)
            for new_label in new_labels:
                f.write("0 " + " ".join(map(str, new_label)) + '\n')