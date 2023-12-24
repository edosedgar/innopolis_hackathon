import warnings
warnings.filterwarnings("ignore")

import os
import glob
import pickle
import argparse
import pandas as pd

import torch

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import predict as sahi_predict

from src.metrics.compute_map import score
from src.utils.submit import parse_labels_v3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='')
    parser.add_argument('weights', type=str,
                        help='')
    parser.add_argument('test_data', type=str,
                        help='')
    parser.add_argument('--csv_dir', type=str, default='submissions',
                        help='')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='')
    parser.add_argument('--imgsz', type=int, default=4000,
                        help='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='')
    parser.add_argument('--overconfident', action='store_true',
                        help='')
    parser.add_argument('--sliced', action='store_true',
                        help='')
    parser.add_argument('--slice_size', type=int, default=640,
                        help='')
    parser.add_argument('--compute_score', action='store_true',
                        help='')
    parser.add_argument('--target_csv', type=str,
                        default='./data/solution_manual_v1.csv',
                        help='')
    
    args = parser.parse_args()
    
    return args


def inference(args):
    device = (
        f"cuda:{args.device}" if torch.cuda.is_available()
        else "cpu"
    )
    model = YOLO(model=args.weights, task='detect')

    test_data_name = os.path.basename(args.test_data.rstrip('/'))
    name = f'{args.model}_{test_data_name}_r{args.imgsz}_t{args.conf}'
    results = model.predict(
        source=args.test_data,
        project=args.output_dir,
        name=name,

        # filtering params
        iou=args.iou,
        conf=args.conf,
        classes=[0],

        # data size
        imgsz=args.imgsz,
        rect=True,

        device=[device],

        # visualization
        show_labels=False,
        show_conf=True,
        show_boxes=True,

        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    return results


def inference_sliced(args):
    device = (
        f"cuda:{args.device}" if torch.cuda.is_available()
        else "cpu"
    )
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.weights,
        device=device
    )

    test_data_name = os.path.basename(args.test_data.rstrip('/'))
    name = f'{args.model}_{test_data_name}_r{args.imgsz}_t{args.conf}'
    results = sahi_predict(
        source=args.test_data,
        project=args.output_dir,
        name=name,

        image_size=args.imgsz,

        model_confidence_threshold=args.conf,
        postprocess_match_metric="IOU", # "IOS" is default
        postprocess_match_threshold=args.iou,

        detection_model=detection_model,
        model_device=device,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        
        # visualization
        novisual=False,
        visual_hide_conf=False,
        visual_hide_labels=True,
        visual_export_format='jpg',
        
        # other
        verbose=1,
        export_pickle=True,
        return_dict=True
    )

    save_dir = results['export_dir']
    # convert pickle to the dir with labels
    labels_dir = os.path.join(save_dir, 'labels')
    pickle_dir = os.path.join(save_dir, 'pickles')
    os.makedirs(labels_dir, exist_ok=True)

    for pkl_name in os.listdir(pickle_dir):
        name = os.path.splitext(pkl_name)[0]
        p_path = os.path.join(pickle_dir, pkl_name)
        l_path = os.path.join(labels_dir, name + '.txt')
        with open(p_path, "rb") as f:
            obj_list = pickle.load(f)

        with open(l_path, "w") as f:
            for i, pred in enumerate(obj_list):
                W, H = 4000, 2250 # TODO
                x, y, w, h = pred.bbox.to_xywh()
                x += w / 2
                y += h / 2
                prob = pred.score.value
                data = [0, x / W, y / H, w / W, h / H, prob]
                print(
                    ("\n" if i != 0 else "") + " ".join(map(str, data)),
                    file=f, end=''
                )

    results = [argparse.Namespace(save_dir=str(save_dir))]
    return results


def save_submission(results, args, target_csv=None):
    save_dir = results[0].save_dir
    print(save_dir)

    df = parse_labels_v3(
        save_dir, overconfident=True,
        target_csv=target_csv
    )
    print(df.head())

    # save submission
    name = os.path.basename(save_dir.rstrip('/'))
    if args.sliced:
        name += f"_sliced{args.slice_size}"
    submission_path = os.path.join(args.csv_dir, 'NeuroEye.csv')

    os.makedirs(args.csv_dir, exist_ok=True)
    df.to_csv(os.path.join(args.csv_dir, f'{name}.csv'), index=False)
    df.to_csv(os.path.join(args.csv_dir, 'NeuroEye.csv'), index=False)    
    return submission_path


if __name__ == "__main__":
    args = parse_args()

    # run inference
    if args.sliced:
        results = inference_sliced(args)
    else:
        results = inference(args)

    # create submission a file
    submission_path = save_submission(results, args, target_csv=args.target_csv)

    # compute approximate score
    if args.compute_score:
        print("Score:", score(
            pd.read_csv(args.target_csv),
            pd.read_csv(submission_path),
            row_id_column_name='file_name'
        ))
    print("Finished")
