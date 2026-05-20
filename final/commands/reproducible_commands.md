# 可复现实验命令

## 生成 Ultralytics 数据配置

```bash
uv run python scripts/prepare_ultralytics_data.py \
  --prepared-root /home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10 \
  --container-path /home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10
```

## 本地 CPU YOLO 继续训练

```bash
uv run --extra yolo yolo detect train \
  model=/home/lh/projects/AeroTrack/runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/weights/best.pt \
  data=/home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10/ultralytics/yolo_data.yaml \
  imgsz=256 epochs=30 batch=4 device=cpu \
  project=/home/lh/projects/AeroTrack/runs/yolo_final_demo \
  name=carrada_ra_cpu10_yolov8n_e30_cpu exist_ok=True
```

## 继续训练

```bash
uv run --extra yolo yolo detect train \
  model=/home/lh/projects/AeroTrack/runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu/weights/best.pt \
  data=/home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10/ultralytics/yolo_data.yaml \
  imgsz=256 epochs=60 batch=4 device=cpu \
  project=/home/lh/projects/AeroTrack/runs/yolo_final_demo \
  name=carrada_ra_cpu10_yolov8n_e60_cpu exist_ok=True
```

## YOLO 展示预测

```bash
uv run --extra yolo yolo detect predict \
  model=/home/lh/projects/AeroTrack/runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu/weights/best.pt \
  source=/home/lh/projects/AeroTrack/runs/yolo_final_demo/carrada_ra_cpu10_showcase_sources.txt \
  imgsz=256 conf=0.001 save=True save_txt=True save_conf=True device=cpu \
  project=/home/lh/projects/AeroTrack/runs/yolo_final_demo \
  name=carrada_ra_cpu10_showcase_pred exist_ok=True
```

## 生成 final 展示包

```bash
uv run python scripts/build_final_showcase.py
```
