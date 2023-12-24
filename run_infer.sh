python infer_yolo.py \
  yolov8x \
  /home/ekaziak1/test_yolo/runs/detect/yolov8x_insulators_100_ep_2_ds_version/weights/best.pt \
  /home/ekaziak1/datasets/test_public/ \
  --csv_dir=./submissions/ \
  --output_dir=./predictions/ \
  --iou=0.5 --conf=0.5 --imgsz=4000 \
  --device=0 \
  --overconfident \
  --compute_score \
  --sliced --slice_size=640 \
