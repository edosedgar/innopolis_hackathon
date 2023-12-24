python infer_yolo.py \
  yolov8x \
  /home/ekaziak1/kopoden/innopolis_hackathon/runs/yolov8x_adamw_v2_complete_ds_v2_r640_run1/weights/best.pt \
  /home/ekaziak1/datasets/test_public_v2/ \
  --target_csv='~/datasets/test_public_v2/manual.csv' \
  --csv_dir=./submissions/ \
  --output_dir=./predictions/ \
  --imgsz=4000 \
  --filtering=nmm --filt_meas=ios --iou=0.5 --conf=0.5 --nmm_shrink=0.1 \
  --device=1 \
  --compute_score \
  --overconfident \
  --sliced \
  --slice_size=640 --slice_overlap=0.2 \
