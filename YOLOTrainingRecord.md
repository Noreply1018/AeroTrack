# YOLO 训练与服务器执行记录

本文档用于记录 AeroTrack 在服务器上进行 YOLO 训练、推理和回传评估时的关键信息。它同时作为本地预处理清单、服务器执行手册和结果表格模板，避免租赁 GPU 服务器后再临时排查数据、配置和输出格式问题。

## 1. 当前结论

当前本地已经适合完成 CARRADA Range-Angle 数据转换、`gt_bbox` 诊断检测、SORT 跟踪、基础指标、失败样例和可视化归档。GPU 服务器应只承担真实 YOLO 训练或 GPU 推理，不应承担数据结构探索、标签调试、SORT 参数 sweep、文档整理或展示图生成。

当前代码状态下，主实验管线仍只支持 `detector.source=gt_bbox` 的 smoke 闭环。`yolo_pretrained` 配置已存在，但 YOLO 推理 adapter 尚未接入 `src/aerotrack/pipeline.py`，因此服务器训练结束后，必须先把 YOLO 原生预测结果转换为项目统一的 `detections.csv`，或者补齐 YOLO adapter 后再接入现有评估链路。

## 2. 本地预处理目标

服务器启动前应在本地完成以下事项：

1. 固定实验数据规模、split、类别映射和归一化规则。
2. 生成 YOLO 可读取的 `images/`、`labels/`、`splits/` 和 `classes.yaml`。
3. 抽查 GT 标注可视化，确认坐标、类别和空帧处理没有明显错误。
4. 跑通 `gt_bbox -> SORT -> metrics -> visualization` 链路。
5. 明确服务器产物必须回传的目录和文件。
6. 准备 YOLO 训练结果记录表，训练完成后立即填写，避免遗失参数和环境信息。

## 3. 推荐本地配置

| 用途 | 配置文件 | 数据规模 | 输出目录 | 说明 |
|---|---|---:|---|---|
| 最小 smoke 验证 | `configs/dataset/carrada_ra_smoke.yaml` | 2 条 sequence | `data/processed/carrada_ra_smoke` | 快速验证数据转换和链路正确性 |
| CPU 扩展验证 | `configs/dataset/carrada_ra_cpu10.yaml` | 10 条 sequence | `data/processed/carrada_ra_cpu10` | 已用于生成当前 CPU 可展示材料 |
| 服务器前置子集 | `configs/dataset/carrada_ra_server30.yaml` | 30 条 sequence，12666 张图片，8750 条标注 | `data/processed/carrada_ra_server30` | 已完成本地预处理，建议作为第一轮 GPU 训练/推理前的数据包 |

对应实验配置：

| 用途 | 配置文件 | 检测来源 | 说明 |
|---|---|---|---|
| 本地诊断闭环 | `configs/experiment/carrada_ra_gtbbox_sort_server30.yaml` | `gt_bbox` | 验证 30 条 sequence 的后处理链路 |
| 服务器 YOLO 推理闭环 | `configs/experiment/carrada_ra_yolopretrained_sort_server30.yaml` | `yolo_pretrained` | adapter 接入后用于统一评估 YOLO 预测 |

## 4. 本地执行清单

建议在租赁服务器前执行：

```bash
uv run pytest
uv run aerotrack run-experiment --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml --preflight-only
uv run aerotrack prepare-data --config configs/experiment/carrada_ra_gtbbox_sort_server30.yaml
uv run aerotrack run-experiment --config configs/experiment/carrada_ra_gtbbox_sort_server30.yaml
```

若 30 条 sequence 处理耗时过长，可先只执行 `prepare-data`，确认 `sample_index.csv`、`annotations.csv`、`labels/` 和 `visual_checks/gt/` 已生成，再决定是否完整跑 `gt_bbox` 实验。

## 5. 本地抽查项

| 检查项 | 文件或目录 | 通过标准 |
|---|---|---|
| 样本索引 | `data/processed/carrada_ra_server30/sample_index.csv` | 存在且行数大于 0 |
| 内部标注 | `data/processed/carrada_ra_server30/annotations.csv` | 存在且标注框坐标有效 |
| YOLO 标签 | `data/processed/carrada_ra_server30/labels/` | 与 `images/` 中样本一一对应 |
| 类别定义 | `data/processed/carrada_ra_server30/classes.yaml` | 类别顺序为 pedestrian、cyclist、car |
| 数据划分 | `data/processed/carrada_ra_server30/splits/*.txt` | train、val、test 文件存在 |
| GT 可视化 | `data/processed/carrada_ra_server30/visual_checks/gt/` | 框位置和类别肉眼检查合理 |
| 诊断检测 | `runs/carrada_ra_gtbbox_sort_server30/detections/detections.csv` | 字段符合统一检测契约 |
| SORT 跟踪 | `runs/carrada_ra_gtbbox_sort_server30/tracks/tracks.csv` | 字段符合统一跟踪契约 |
| 指标归档 | `runs/carrada_ra_gtbbox_sort_server30/metrics/summary.csv` | 可直接汇总进展示表 |

当前本地检查结果：

| 项目 | 结果 |
|---|---:|
| server30 总图片数 | 12666 |
| server30 总标注框数 | 8750 |
| train 图片数 | 8088 |
| val 图片数 | 2448 |
| test 图片数 | 2130 |
| GT 可视化检查图 | 20 |
| 诊断检测行数 | 8750 |
| SORT 跟踪行数 | 8750 |
| 实验可视化文件数 | 2401 |

## 6. 同步到服务器的内容

建议只同步必要文件，避免上传原始 136 GiB CARRADA 数据。

必须同步：

```text
configs/
src/
scripts/
pyproject.toml
uv.lock
data/processed/carrada_ra_server30/
```

可选同步：

```text
runs/carrada_ra_gtbbox_sort_server30/config.yaml
runs/carrada_ra_gtbbox_sort_server30/metrics/summary.csv
runs/carrada_ra_gtbbox_sort_server30/visualizations/
```

不建议同步：

```text
data/carrada/
runs/*/visualizations/ 中大量非必要图片
.venv/
.pytest_cache/
```

## 7. 服务器环境检查

服务器启动后先安装包含 YOLO 的可选依赖，再执行检查。`ultralytics` 属于项目的 `yolo` optional dependency，不在默认依赖中。

```bash
uv sync --extra yolo --extra tracking --extra vision
```

随后执行：

```bash
nvidia-smi
python --version
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import ultralytics; print(ultralytics.__version__)"
```

记录到下表：

| 项目 | 值 |
|---|---|
| 服务器供应商 | 按实际租赁实例填写 |
| 服务器实例规格 | 推荐单卡 NVIDIA RTX 4090 / L40S / A5000 及以上 |
| GPU 型号 | 按 `nvidia-smi` 实测填写 |
| GPU 数量 | 推荐 1 |
| 显存 | 推荐 >= 16 GiB，优先 24 GiB |
| CPU | 推荐 >= 8 vCPU |
| 内存 | 推荐 >= 32 GiB |
| 磁盘 | 推荐 >= 100 GiB SSD |
| 操作系统 | 推荐 Ubuntu 22.04 LTS |
| Python 版本 | 推荐 3.11 |
| PyTorch 版本 | 按 `uv run python -c "import torch; print(torch.__version__)"` 实测填写 |
| CUDA 版本 | 按 `nvidia-smi` / PyTorch 实测填写 |
| Ultralytics 版本 | 按 `uv run python -c "import ultralytics; print(ultralytics.__version__)"` 实测填写 |
| AeroTrack Git commit | 服务器执行时按 `git rev-parse --short HEAD` 实测填写 |

## 8. YOLO 训练配置记录

每次训练必须记录。当前 `configs/detector/yolo_train.yaml` 是 Stage1 保留模板，默认仍指向 smoke 数据和 `weights/yolo_pretrained.pt`，且关闭自动下载。服务器首轮训练建议按下表作为覆盖配置执行：先生成 Ultralytics 数据配置，再把 YOLOv8n 预训练权重放到 `weights/yolo_pretrained.pt`，并显式设置 `project=runs`、`name=carrada_ra_yolov8n_server30_baseline`。

生成服务器可用数据配置：

```bash
uv run python scripts/prepare_ultralytics_data.py \
  --prepared-root data/processed/carrada_ra_server30 \
  --container-path /workspace/data/processed/carrada_ra_server30
```

| 字段 | 值 |
|---|---|
| 实验名称 | `carrada_ra_yolov8n_server30_baseline` |
| 训练日期 | 2026-05-20 计划配置，服务器执行后按实际日期更新 |
| YOLO 实现 | Ultralytics |
| YOLO 模型版本 | YOLOv8n |
| 初始权重 | `weights/yolo_pretrained.pt`，内容建议使用 YOLOv8n 预训练权重 |
| 数据配置 | `data/processed/carrada_ra_server30/ultralytics/yolo_data.yaml` |
| 输入表示 | CARRADA Range-Angle PNG |
| 类别数 | 3 |
| 类别名称 | pedestrian、cyclist、car |
| 训练 sequence 数 | 18 |
| 验证 sequence 数 | 6 |
| 测试 sequence 数 | 6 |
| 训练图片数 | 8088 |
| 验证图片数 | 2448 |
| 测试图片数 | 2130 |
| 标注框总数 | 8750 |
| 图像尺寸 `imgsz` | 640 |
| epoch | 50 |
| batch | 8，若显存 >= 24 GiB 可尝试 16 |
| optimizer | AdamW |
| learning rate | 0.001 |
| conf 阈值 | 0.25 |
| NMS IoU 阈值 | 0.70 |
| 训练耗时 | 服务器训练后填写 |
| 最佳权重 | `runs/carrada_ra_yolov8n_server30_baseline/weights/best.pt` |
| 最终权重 | `runs/carrada_ra_yolov8n_server30_baseline/weights/last.pt` |
| 备注 | 首轮推荐使用 YOLOv8n 做轻量 baseline，优先验证标签质量、训练链路和后处理接入；稳定后再扩展到 YOLOv8s/YOLOv11n 或全量数据。 |

## 9. 工程诊断结果表

本表记录 `gt_bbox` 诊断闭环结果，只能证明数据、统一检测格式、SORT、评估和可视化链路可运行。这里的检测指标来自 GT 框转检测框，不代表 YOLO 模型性能，不能作为正式 YOLO baseline 展示。

| 实验名 | 检测来源 | 数据集 | train/val/test 图片数 | 标注框数 | 诊断 precision | 诊断 recall | 诊断 F1 | 诊断 mAP50 | SORT MOTA | IDF1 状态 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| carrada_ra_gtbbox_sort_smoke | GT bbox | carrada_ra_smoke | 1017/0/264 | 436 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.520000 | unavailable | 最小诊断闭环，不是 YOLO baseline |
| carrada_ra_gtbbox_sort_cpu10 | GT bbox | carrada_ra_cpu10 | 4495/704/528 | 2425 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.462810 | unavailable | CPU 扩展诊断闭环 |
| carrada_ra_gtbbox_sort_server30 | GT bbox | carrada_ra_server30 | 8088/2448/2130 | 8750 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.411548 | unavailable | 服务器前置子集诊断闭环 |

## 10. 真实 YOLO 训练结果表

本表只填写真实 YOLO 训练或推理结果。服务器训练完成前，正式 server30 baseline 指标保持待填写。本地 `cpu10` 短训结果只用于展示链路和产物，不作为最终精度 baseline。

| 实验名 | 模型版本 | 初始权重 | 数据集 | train/val/test 图片数 | 标注框数 | imgsz | epoch | batch | precision | recall | F1 | mAP50 | mAP50-95 | SORT MOTA | IDF1 状态 | 训练耗时 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| carrada_ra_cpu10_yolov8n_cpu | YOLOv8n | `yolov8n.pt` | carrada_ra_cpu10 | 4495/704/528 | 2425 | 256 | 3 | 4 | 0.00170 | 0.08257 | 低指标未单独汇总 | 0.00122 | 0.00019 | 未接入 SORT | unavailable | 0.276 hours | 本地 CPU 短训已完成，产物见 `runs/yolo_local_demo/carrada_ra_cpu10_yolov8n_cpu/`，展示报告见 `docs/YOLOLocalDemoReport.md`；仅用于证明 YOLO 训练/验证/可视化链路可跑通 |
| carrada_ra_yolov8n_server30_baseline | YOLOv8n | `weights/yolo_pretrained.pt` | carrada_ra_server30 | 8088/2448/2130 | 8750 | 640 | 50 | 8 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | 训练后填写 | unavailable | 训练后填写 | 首轮真实 YOLO baseline 推荐配置；使用 `data/processed/carrada_ra_server30/ultralytics/yolo_data.yaml`，并显式设置 `project=runs`、`name=carrada_ra_yolov8n_server30_baseline`；指标必须来自服务器训练/验证结果，不可用 GT 诊断指标替代 |

## 11. 服务器产物回传清单

Ultralytics 默认输出通常在 `runs/detect/train*` 或用户通过 `project/name` 指定的目录中。下面列出两类路径：原始默认路径用于定位文件，项目归档路径用于回传后整理。

Ultralytics 原始默认路径示例：

```text
runs/detect/train*/weights/best.pt
runs/detect/train*/weights/last.pt
runs/detect/train*/args.yaml
runs/detect/train*/results.csv
runs/detect/train*/results.png
runs/detect/train*/confusion_matrix.png
runs/detect/train*/val_batch*_pred.jpg
```

建议整理后的项目归档路径：

```text
runs/<yolo_experiment_name>/weights/best.pt
runs/<yolo_experiment_name>/weights/last.pt
runs/<yolo_experiment_name>/args.yaml
runs/<yolo_experiment_name>/results.csv
runs/<yolo_experiment_name>/results.png
runs/<yolo_experiment_name>/confusion_matrix.png
runs/<yolo_experiment_name>/val_batch*_pred.jpg
runs/<aerotrack_experiment_name>/detections/detections.csv
```

若服务器直接跑完 AeroTrack 后处理，还应拉回：

```text
runs/<aerotrack_experiment_name>/tracks/tracks.csv
runs/<aerotrack_experiment_name>/metrics/detection_metrics.json
runs/<aerotrack_experiment_name>/metrics/tracking_metrics.json
runs/<aerotrack_experiment_name>/metrics/summary.csv
runs/<aerotrack_experiment_name>/visualizations/
```

## 12. 后处理接回本地

服务器回传 `detections.csv` 后，本地应复用现有 SORT、评估和可视化链路。但当前 CLI 会先执行 `yolo_pretrained` preflight，若本地没有 `weights/yolo_pretrained.pt`，即使 `detections.csv` 已存在也会被权重 gate 拦住。因此有两种做法：

1. 把服务器回传的 YOLO 权重放到 `weights/yolo_pretrained.pt`，再执行分阶段命令。
2. 后续补一个“跳过检测器权重检查、只消费现有 detections.csv”的配置或 CLI 选项，再执行分阶段命令。

当前可执行方式是先准备权重：

```bash
mkdir -p weights
cp <server_best_or_exported_weight>.pt weights/yolo_pretrained.pt
```

然后执行：

```bash
uv run aerotrack run-tracking --config configs/experiment/carrada_ra_yolopretrained_sort_server30.yaml
uv run aerotrack evaluate --config configs/experiment/carrada_ra_yolopretrained_sort_server30.yaml
uv run aerotrack visualize --config configs/experiment/carrada_ra_yolopretrained_sort_server30.yaml
```

注意：上述命令依赖 YOLO adapter 或服务器回传的统一 `detections.csv` 已放到目标实验目录。若 adapter 尚未实现，应先完成 `YOLO 原生输出 -> detections.csv` 的转换；若本地没有权重文件，当前 preflight 仍会 gate。

## 13. 风险与边界

1. `gt_bbox` 结果只用于诊断工程链路，不能作为正式 YOLO baseline。
2. 当前 IDF1、ID switches、track fragmentation 仍标记为 unavailable，不能伪造成有效身份指标。
3. 服务器训练前必须确认 label 坐标和类别映射，否则训练结果不可解释。
4. 第一轮 GPU 训练建议先使用 30 条 sequence 的子集，确认流程稳定后再扩大数据量。
5. 每次训练必须记录 Git commit、数据配置、模型版本和权重路径，否则后续无法复现实验。
