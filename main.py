import os
import argparse


class Config:
    # models params
    model = 'yolov8x'
    weights = os.path.join('models', 'best.pt')

    # inference params
    iou = 0.5
    conf = 0.5
    imgsz = 4000

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
    cmd = f"""
      python infer_yolo.py \
        {Config.model} \
        {Config.weights} \
        {args.data_dir} \
        --target_csv='~/datasets/test_public_v2/manual.csv' \
        --csv_dir=./submissions/ \
        --output_dir=./predictions/ \
        --iou={Config.iou} --conf={Config.conf} --imgsz={Config.imgsz} \
        --device=0 \
        --overconfident \
        --compute_score \
        --sliced \
        --slice_size=640 \
    """
    os.system(cmd)


if __name__ == "__main__":
    main()
