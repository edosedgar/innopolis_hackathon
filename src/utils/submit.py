import os
import pandas as pd


def labels_to_submission(data_dir, overconfident=False, target_csv=''):
    df = pd.DataFrame(columns=['file_name', 'rbbox', 'probability'])

    target_names = None
    if target_csv:
        target_df = pd.read_csv(target_csv)
        target_names = target_df.file_name.values

    img_dir = data_dir
    if os.path.exists(os.path.join(data_dir, 'visuals')):
        img_dir = os.path.join(data_dir, 'visuals')

    for file in os.listdir(img_dir):
        if not file.endswith(('.JPG', '.jpg')):
            continue
        filename = os.path.splitext(file)[0]
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

        if len(boxes) == 0: # Edgar: dont continue, add zeros
            boxes = [[0, 0, 0, 0]]
            probs = [0]
        df.loc[len(df)] = [filename, boxes, probs]

    return df
