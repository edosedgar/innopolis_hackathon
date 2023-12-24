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
    parser.add_argument('--imgsz', type=int, default=640,
                        help='')
    parser.add_argument('--amp', type=int, default=1)
    
    args = parser.parse_args()
    return args


def train(args):
    # load configs
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    data_cfg = EasyDict(data_cfg)
    print('Model: ', cfg.base_model_weights)
    print('Epochs: ', cfg.train_params.epochs)
    print('Batch / Full batch: ', cfg.train_params.batch, '/', cfg.train_params.nbs)

    # initialize mode
    model = YOLO(cfg.base_model_weights)

    # run training
    test_data_name = os.path.splitext(os.path.basename(args.data_config))[0]
    name = os.path.splitext(os.path.basename(args.config))[0]
    name += f"_{test_data_name}"
    name += f"_r{args.imgsz}"
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
        device=0,
        verbose=False,
        
        # training params from the config
        **cfg.train_params
    )
    print("Finished")


if __name__ == "__main__":
    args = parse_args()
    train(args)
