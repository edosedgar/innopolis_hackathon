import warnings
warnings.filterwarnings("ignore")

import os
import yaml
import argparse
from easydict import EasyDict

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='')
    parser.add_argument('data_config', type=str, help='')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='')
    parser.add_argument('--run_id', type=str, default='1',
                        help='helpful for training with the same config and and data')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='')
    parser.add_argument('--bs', type=int, default=None)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--amp', type=int, default=1)

    args = parser.parse_args()
    if args.device.lower() in ['', 'cpu']:
        args.device = 'cpu'
    else:
        args.device = list(map(int, args.device.strip(" \n\r\t,").split(',')))
    return args


def train(args):
    # load configs
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    data_cfg = EasyDict(data_cfg)

    if args.bs is not None:
        cfg.train_params.batch = args.bs
    cfg.train_params.batch *= (1 if args.device == "cpu" else len(args.device))
    print('Model: ', cfg.base_model_weights)
    print('Epochs: ', cfg.train_params.epochs)
    print('Batch / Full batch: ', cfg.train_params.batch, '/', cfg.train_params.nbs)

    # initialize mode
    model = YOLO(cfg.base_model_weights)

    # run training
    test_data_name = os.path.splitext(os.path.basename(args.data_config))[0]
    name = os.path.splitext(os.path.basename(args.config))[0]
    name += f"_{test_data_name}_r{args.imgsz}_run{args.run_id}"
    model.train(
        data=args.data_config,

        task='detect',
        imgsz=args.imgsz,
        fraction=1.0,

        project=args.output_dir,
        name=name,

        val=True,
        amp=bool(args.amp),
        exist_ok=True,
        resume=False,
        device=args.device,
        verbose=False,

        # training params from the config
        **cfg.train_params
    )
    print("Finished")


if __name__ == "__main__":
    args = parse_args()
    train(args)
