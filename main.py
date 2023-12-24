import os
import argparse


class Config:
    # models params
    model = 'yolov8x'
    weights = os.path.join(
        'models', 'yolov8x_best',
        'weights', 'best.pt'
    )

    # inference params
    iou = 0.7
    conf = 0.7
    imgsz = 4000

    # sliced inference
    sliced = True
    slice_size = 640
    nmm_shrink = 0.1
    filtering = 'nmm' # non-max merging
    filt_meas = 'ios' # intersection over smaller area

    # leaderboard tricks
    overconfident = False

    # run params
    device = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str,
                        help='path to the directory with test images')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cmd = (
        "python infer_yolo.py "
        f"{Config.model} "
        f"{Config.weights} "
        f"{args.data_dir} "
        f"--csv_dir=submissions "
        f"--output_dir=predictions "
        "--target_csv='' "
        f"--iou={Config.iou} --conf={Config.conf} --imgsz={Config.imgsz} "
        f"--device={Config.device} "
    )
    if Config.sliced:
        cmd += f"--filtering={Config.filtering} --filt_meas={Config.filt_meas} "
        cmd += f"--sliced --slice_size={Config.slice_size} --nmm_shrink={Config.nmm_shrink} "
    if Config.overconfident:
        cmd += "--overconfident "
    os.system(cmd)


if __name__ == "__main__":
    main()
