python infer_yolo.py \
  yolov8x \
  models/yolov8x_best/weights/model.pt \
  datasets/test_public_v2/ \
  --target_csv=datasets/test_public_v2/manual.csv \
  --csv_dir=./submissions/ \
  --output_dir=./predictions/ \
  --imgsz=4000 \
  --filtering=nmm --filt_meas=ios --iou=0.7 --conf=0.7 --nmm_shrink=0.1 \
  --device=0 \
  --compute_score \
  --sliced \
  --slice_size=640 --slice_overlap=0.2 \
