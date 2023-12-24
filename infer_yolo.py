import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import shutil
import argparse
import pandas as pd
from PIL import Image

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import predict as sahi_predict

from src.metrics.compute_map import score
from src.utils.submit import labels_to_submission


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
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='')
    parser.add_argument('--filtering', type=str, default='nmm')
    parser.add_argument('--filt_meas', type=str, default='ios')
    parser.add_argument('--overconfident', action='store_true',
                        help='')
    parser.add_argument('--nmm_shrink', type=float, default=0.0)
    parser.add_argument('--sliced', action='store_true',
                        help='')
    parser.add_argument('--slice_overlap', type=float, default=0.2,
                        help='')
    parser.add_argument('--slice_size', type=int, default=640,
                        help='')
    parser.add_argument('--compute_score', action='store_true',
                        help='')
    parser.add_argument('--target_csv', type=str,
                        default='~/datasets/test_public_v2/manual.csv',
                        help='')

    args = parser.parse_args()
    if args.device.lower() in ['', 'cpu']:
        args.device = 'cpu'
    else:
        args.device = int(args.device)

    return args


def inference(name, args):
    model = YOLO(model=args.weights, task='detect')

    ## ultralytics library appends to txt, instead of overwriting
    if os.path.isdir(f'{args.output_dir}/{name}/labels/'):
        for filename in os.listdir(f'{args.output_dir}/{name}/labels/'):
            os.unlink(f'{args.output_dir}/{name}/labels/{filename}')

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

        device=args.device,

        # visualization
        show_labels=False,
        show_conf=True,
        show_boxes=True,
        line_width=1,

        save=True,
        save_txt=True,
        save_conf=True,
        exist_ok=True,
    )
    return results


def inference_sliced(name, args):
    run_dir = os.path.join(args.output_dir, name)
    print(run_dir)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    device = args.device
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.weights,
        device=device,
        image_size=args.imgsz,
        confidence_threshold=args.conf,
    )

    results = sahi_predict(
        source=args.test_data,
        project=args.output_dir,
        name=name,

        postprocess_type={
            "nmm": "GREEDYNMM",
            "nms": "NMS",
        }[args.filtering.lower()],
        postprocess_match_metric={
            "ios": "IOS",
            "iou": "IOU"
        }[args.filt_meas.lower()],
        postprocess_match_threshold=args.iou,

        detection_model=detection_model,
        model_device=device,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.slice_overlap,
        overlap_width_ratio=args.slice_overlap,

        # visualization
        novisual=False,
        visual_hide_conf=False,
        visual_hide_labels=True,
        visual_export_format='jpg',
        visual_bbox_thickness=1,

        # other
        verbose=1,
        export_pickle=True,
        return_dict=True
    )

    save_dir = results['export_dir']
    # convert pickle to the dir with labels
    labels_dir = os.path.join(save_dir, 'labels')
    images_dir = os.path.join(save_dir, 'visuals')
    pickle_dir = os.path.join(save_dir, 'pickles')
    os.makedirs(labels_dir, exist_ok=True)

    for pkl_name in os.listdir(pickle_dir):
        name = os.path.splitext(pkl_name)[0]
        p_path = os.path.join(pickle_dir, pkl_name)
        l_path = os.path.join(labels_dir, name + '.txt')
        i_path = os.path.join(images_dir, name + '.jpg')
        with open(p_path, "rb") as f:
            obj_list = pickle.load(f)

        with open(l_path, "w") as f:
            for i, pred in enumerate(obj_list):
                pil_img = Image.open(i_path)
                W, H = pil_img.size
                x, y, w, h = pred.bbox.to_xywh()
                x += w / 2
                y += h / 2
                prob = pred.score.value
                if args.filtering == 'nmm':
                    w *= (1.0 - args.nmm_shrink)
                    h *= (1.0 - args.nmm_shrink)
                data = [0, x / W, y / H, w / W, h / H, prob]
                print(
                    ("\n" if i != 0 else "") + " ".join(map(str, data)),
                    file=f, end=''
                )

    results = [argparse.Namespace(save_dir=str(save_dir))]
    return results


def save_submission(name, results, args, target_csv=''):
    save_dir = results[0].save_dir
    print(f"Submission saved to {save_dir}")

    df = labels_to_submission(
        save_dir, overconfident=args.overconfident,
        target_csv=target_csv
    )
    print(df.head())

    # save submission
    test_data_name = os.path.basename(args.test_data.rstrip('/'))
    final_dir = os.path.join(args.csv_dir, test_data_name)
    submission_path = os.path.join(final_dir, 'NeuroEye.csv')

    os.makedirs(final_dir, exist_ok=True)
    df.to_csv(os.path.join(final_dir, f'{name}.csv'), index=False)
    df.to_csv(os.path.join(final_dir, 'NeuroEye.csv'), index=False)
    return submission_path


if __name__ == "__main__":
    args = parse_args()

    # run inference
    overconf = "_overconf" if args.overconfident else ""
    sliced = f"_sliced{args.slice_size}" if args.sliced else ""
    filtering = "nms" if not args.sliced else args.filtering
    filt_meas = "iou" if not args.sliced else args.filt_meas
    name = os.path.basename(
        os.path.abspath(
            os.path.join(os.path.dirname(args.weights), '..')
        )
    )
    name += f'_r{args.imgsz}'
    name += f'_{args.filtering}-{filt_meas}{args.iou}'
    name += f'_t{args.conf}{overconf}{sliced}'
    if args.sliced:
        results = inference_sliced(name, args)
    else:
        results = inference(name, args)

    # create submission a file
    submission_path = save_submission(
        name, results, args, target_csv=args.target_csv
    )

    # compute approximate score
    if args.compute_score:
        print("Score (map50):", score(
            pd.read_csv(args.target_csv),
            pd.read_csv(submission_path),
            row_id_column_name='file_name'
        ))
    print("Finished")
