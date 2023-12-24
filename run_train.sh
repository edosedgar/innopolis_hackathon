python train_yolo.py \
  configs/train/yolov8x_adamw_best.yaml \
  configs/data/complete_ds_v2.yaml \
  --output_dir=runs --run_id=1 \
  --imgsz=640 --bs=8 \
  --device=0,1 \
