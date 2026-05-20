# YOLO 本地小规模训练展示报告

## 1. 实验目的

本次实验用于验证 AeroTrack 已经具备将 CARRADA Range-Angle 图像接入 YOLO 检测器的基本能力，并生成可用于答辩展示的训练曲线、验证图、预测图和指标表。

本实验定位为工程链路与展示材料验证，不作为最终高精度 YOLO baseline。原因是训练仅使用 CPU、`3` 个 epoch 和 `carrada_ra_cpu10` 小规模子集，模型尚未充分收敛。

## 2. 数据与环境

| 项目 | 内容 |
|---|---|
| 数据集 | `data/processed/carrada_ra_cpu10` |
| 输入表示 | CARRADA Range-Angle PNG，尺寸 `256x256` |
| 类别 | `pedestrian`、`cyclist`、`car` |
| 训练图片数 | `4495` |
| 验证图片数 | `704` |
| 测试图片数 | `528` |
| 训练框架 | Ultralytics `8.4.48` |
| 模型 | `yolov8n.pt` |
| 运行设备 | CPU |
| Python / Torch | Python `3.11.15`，Torch `2.11.0+cu130`，CUDA 不可用 |

Docker 镜像 `ultralytics/ultralytics:latest` 在本机拉取过程中长时间卡住，因此本次训练改用本地 `uv run --extra yolo` 环境完成。数据配置和训练命令保持与 Docker 方案等价，产物仍可复现。

## 3. 执行命令

生成 Ultralytics 数据配置：

```bash
uv run python scripts/prepare_ultralytics_data.py \
  --prepared-root data/processed/carrada_ra_cpu10 \
  --container-path /home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10
```

训练命令：

```bash
uv run --extra yolo yolo detect train \
  model=yolov8n.pt \
  data=/home/lh/projects/AeroTrack/data/processed/carrada_ra_cpu10/ultralytics/yolo_data.yaml \
  imgsz=256 \
  epochs=3 \
  batch=4 \
  device=cpu \
  project=/home/lh/projects/AeroTrack/runs/yolo_local_demo \
  name=carrada_ra_cpu10_yolov8n_cpu \
  exist_ok=True
```

展示预测命令：

```bash
uv run --extra yolo yolo detect predict \
  model=/home/lh/projects/AeroTrack/runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/weights/best.pt \
  source=/home/lh/projects/AeroTrack/runs/yolo_local_demo/showcase_sources.txt \
  imgsz=256 \
  conf=0.001 \
  save=True \
  save_txt=True \
  save_conf=True \
  device=cpu \
  project=/home/lh/projects/AeroTrack/runs/yolo_local_demo \
  name=carrada_ra_cpu10_showcase_pred \
  exist_ok=True
```

## 4. 结果表

| epoch | train box loss | train cls loss | train dfl loss | precision | recall | mAP50 | mAP50-95 | val box loss | val cls loss | val dfl loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.44543 | 5.33872 | 1.33387 | 0.00012 | 0.03670 | 0.00000 | 0.00000 | 1.16787 | 7.84747 | 0.53627 |
| 2 | 2.50865 | 4.13320 | 1.27943 | 0.00027 | 0.06422 | 0.00002 | 0.00000 | 1.17596 | 4.20154 | 0.54865 |
| 3 | 2.46800 | 3.39530 | 1.25853 | 0.00170 | 0.08257 | 0.00122 | 0.00019 | 1.24137 | 3.10018 | 0.57813 |

结论：

1. 训练已完整跑完 `3` 个 epoch，并生成 `best.pt` 与 `last.pt`。
2. 分类损失从 `5.33872` 降到 `3.39530`，说明训练链路有效，模型参数发生了学习。
3. 检测指标仍很低，说明当前 CPU 小规模短训只能用于展示链路，不适合作为正式精度结果。

## 5. 可展示产物

训练与验证图表：

| 用途 | 文件 |
|---|---|
| 训练曲线总览 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/results.png` |
| PR 曲线 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/BoxPR_curve.png` |
| F1 曲线 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/BoxF1_curve.png` |
| 混淆矩阵 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/confusion_matrix.png` |
| 标签分布 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/labels.jpg` |

验证样例图：

| 用途 | 文件 |
|---|---|
| 验证 GT 第 1 批 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/val_batch0_labels.jpg` |
| 验证预测第 1 批 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/val_batch0_pred.jpg` |
| 验证 GT 第 2 批 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/val_batch1_labels.jpg` |
| 验证预测第 2 批 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/val_batch1_pred.jpg` |

展示预测图：

| 用途 | 文件 |
|---|---|
| 低阈值预测样例 1 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000029.jpg` |
| 低阈值预测样例 2 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000042.jpg` |
| 低阈值预测样例 3 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000043.jpg` |
| 低阈值预测样例 4 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000045.jpg` |
| 低阈值预测样例 5 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000046.jpg` |
| 低阈值预测样例 6 | `runs/yolo_local_demo/carrada_ra_cpu10_showcase_pred/000047.jpg` |

权重与原始指标：

| 用途 | 文件 |
|---|---|
| 最佳权重 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/weights/best.pt` |
| 最终权重 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/weights/last.pt` |
| 原始训练指标 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/results.csv` |
| 训练参数 | `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/args.yaml` |

## 6. 答辩表述建议

建议表述：

> 本项目已经完成 CARRADA Range-Angle 图像到 YOLO 格式的转换，并用 YOLOv8n 在本地环境完成了小规模训练、验证和预测可视化。该实验用于证明雷达 RA 图可以接入通用视觉检测框架，并产出可复现的训练曲线、验证图和检测框示例。

需要避免的表述：

> 当前模型已经达到可用精度。

更严谨的说法：

> 当前结果是 CPU 小规模短训产物，检测指标较低，主要用于展示工程链路。若要形成正式 YOLO baseline，需要使用 GPU、更多 epoch、更完整数据集，并将 YOLO 输出接入 AeroTrack 统一 `detections.csv`、SORT 和 metrics 流程。

## 7. 后续改进

1. 在 GPU 环境下使用 `server30` 或更大数据子集训练更多 epoch。
2. 实现 YOLO 预测结果到 AeroTrack 统一 `detections.csv` 的转换。
3. 将 YOLO 检测结果接入 SORT，生成 `tracks.csv` 和跟踪可视化。
4. 用统一评估口径补充正式检测指标和跟踪指标。
