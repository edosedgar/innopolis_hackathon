import os
from glob import glob
import pandas as pd

from src.metrics.compute_map import score


def parse_labels(label_dir, save: bool = False, save_path: str = ''):
    data_list = []
    for fname in os.listdir(label_dir):
        full_path = os.path.join(label_dir, fname)
        with open(full_path, "r") as f:
            lines = f.readlines()
            boxes = [l.strip('\n\r\t').split(' ') for l in lines]
        key = os.path.splitext(fname)[0]
        for i, box in enumerate(boxes, start=1):
            data_list.append({
                'file_name': key + f'_{i}',
                'x': box[1], 'y': box[2], 'w': box[3], 'h': box[4],
                'probability': box[5]
            })

    columns = ['file_name', 'x', 'y', 'w', 'h', 'probability']
    submission = pd.DataFrame(data=data_list, columns=columns)

    if save:
        submission.to_csv(save_path, index=False)
    return submission


def parse_labels_v2(label_dir, save: bool = False, save_path: str = ''):
    data_list = []
    for fname in os.listdir(label_dir):
        full_path = os.path.join(label_dir, fname)
        with open(full_path, "r") as f:
            lines = f.readlines()
            boxes = [l.strip('\n\r\t').split(' ') for l in lines]
        key = os.path.splitext(fname)[0]
        for i, box in enumerate(boxes, start=1):
            data_list.append({
                'file_name': key,
                'x': box[1], 'y': box[2], 'w': box[3], 'h': box[4],
                'probability': box[5]
            })

    columns = ['file_name', 'x', 'y', 'w', 'h', 'probability']
    submission = pd.DataFrame(data=data_list, columns=columns)

    # for some reason yolo duplicates detections 3 times, remove duplicates
    submission.drop_duplicates(
        subset=submission.columns.values[1:], keep='first', inplace=True
    )

    # reorganize to match submission type-2
    grouped = submission.groupby(['file_name']) \
                        .aggregate(lambda x: ','.join(x))\
                        .reset_index() \
                        .set_index('file_name')

    # insert in submission file
    template_sub = pd.read_csv(
        './submissions/solution.csv',
        index_col=0
    )
    for key in template_sub.index.values:
        if key in grouped.index.values:
            template_sub.loc[key] = grouped.loc[key]

    # make file_name a non-index column again
    template_sub = template_sub.reset_index()

    if save:
        template_sub.to_csv(save_path, index=False)
    return template_sub


def parse_labels_v3(data_dir, overconfident=False, target_csv=None):
    files = sorted(glob(os.path.join(data_dir, "*.JPG")))
    if len(files) == 0:
        files = sorted(glob(os.path.join(data_dir, 'visuals', "*.jpg")))
    df = pd.DataFrame(columns=['file_name', 'rbbox', 'probability'])

    target_names = None
    if target_csv:
        target_df = pd.read_csv(target_csv)
        target_names = target_df.file_name.values

    for file in files:
        filename = os.path.splitext(os.path.basename(file))[0]
        txt_file = os.path.join(data_dir, 'labels', filename + '.txt')

        # skip files not mentioned in target csv
        if target_names is not None and filename not in target_names:
            continue

        try:
            f = open(txt_file)
            boxes = []
            probs = []
            for line in f:
                _, x, y, w, h, p = map(float, line.split(" "))
                if overconfident:
                    p = 1
                boxes.append([x, y, w, h])
                probs.append(p)
        except:
            boxes = [[0, 0, 0, 0]]
            probs = [0]

        # for i in range(len(lines)):
        #     lines[i][0] = lines[i][0]
        #     lines[i][1] = lines[i][1]
        if len(boxes) == 0: # Edgar: dont continue, add zeros
            boxes = [[0, 0, 0, 0]]
            probs = [0]
        df.loc[len(df)] = [filename, boxes, probs]

    # df.to_csv("NeuroEye.csv", index=False)
    return df
