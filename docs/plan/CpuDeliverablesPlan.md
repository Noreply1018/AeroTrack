# CPU 环境可产出材料计划

本文档沉淀当前本机 CPU 环境下可以真实产出的 AeroTrack 项目材料。它的目标是把“无需真实 YOLO 训练也能完成的产物”和“当前环境下不能直接完成的产物”分清楚，避免把诊断闭环包装成正式 YOLO baseline。

## 1. 当前环境审计结论

当前本机适合完成数据转换、`gt_bbox` 诊断检测、SORT 跟踪、基础评估、可视化和实验归档；不适合完成真实 YOLO 训练。

已审计到的本机条件：

1. CPU：12th Gen Intel Core i7-12700H，20 线程。
2. 内存：约 7.6 GiB，总可用约 4.4 GiB。
3. 磁盘：约 800 GiB 可用。
4. GPU：当前 Linux 环境未发现 `nvidia-smi`，按无可用 GPU 处理。
5. Python 依赖：`numpy`、`PIL`、`scipy`、`motmetrics`、`pyyaml` 可用。
6. YOLO 依赖：当前未安装 `torch` 和 `ultralytics`。
7. YOLO 权重：当前未发现 `weights/yolo_pretrained.pt`。

本机已有数据状态：

1. 原始 CARRADA 数据目录：`data/carrada/Carrada`。
2. 原始数据体量：`data/carrada` 约 136 GiB。
3. 原始 Range-Angle `.npy` 数量：约 12666 帧。
4. 已处理 smoke 数据目录：`data/processed/carrada_ra_smoke`，约 30 MiB。
5. 已处理 smoke 样本数：`sample_index.csv` 约 1281 条样本。
6. 已处理标注数：`annotations.csv` 约 436 条标注。
7. 已有 smoke 实验目录：`runs/carrada_ra_gtbbox_sort_smoke`。

## 2. 当前可以真实拿到的产物

### 2.1 雷达数据转换产物

这些产物不依赖 YOLO 训练，也不依赖 GPU。

可产出内容：

1. Range-Angle PNG 图像。
2. YOLO label 文本文件。
3. `sample_index.csv`。
4. `annotations.csv`。
5. `conversion_records.csv`。
6. `classes.yaml`。
7. `splits/train.txt`、`splits/val.txt`、`splits/test.txt`。
8. GT 标注框叠加在 Range-Angle 图上的可视化检查图。

展示价值：

1. 证明 CARRADA Range-Angle 数据已经被转换成检测模型可读取的数据组织形式。
2. 证明坐标、类别和标注框来源有统一记录。
3. 证明后续 YOLO 或其他检测器可以复用同一套数据接口。

验收方式：

1. 检查 `data/processed/carrada_ra_smoke/sample_index.csv` 是否存在且有样本。
2. 检查 `data/processed/carrada_ra_smoke/annotations.csv` 是否存在且有标注。
3. 检查 `data/processed/carrada_ra_smoke/images/` 和 `labels/` 是否一一对应。
4. 检查 `data/processed/carrada_ra_smoke/visual_checks/gt/` 是否能看到标注叠加图。

### 2.2 `gt_bbox` 诊断检测产物

`gt_bbox` 不是正式检测模型，它的作用是把 GT 标注框转换成统一检测结果，用于验证下游跟踪、评估和可视化链路。

可产出内容：

1. `runs/carrada_ra_gtbbox_sort_smoke/detections/detections.csv`。
2. detection source 记录为 `gt_bbox` 的实验配置。
3. 检测框可视化图。
4. 基于 GT 检测框得到的检测指标。

展示价值：

1. 证明项目内部统一检测结果格式可用。
2. 证明下游模块不依赖 YOLO 私有输出，而是读取统一 `detections.csv`。
3. 证明跟踪、评估和可视化模块可以在稳定输入下跑通。

边界说明：

1. `gt_bbox` 指标通常会接近理想检测结果，不能作为 YOLO 模型效果。
2. `gt_bbox` 只能作为诊断闭环和上限参考，不能包装成正式 YOLO baseline。

### 2.3 SORT 跟踪产物

当前 CPU 环境可以基于 `gt_bbox` 诊断检测结果运行 SORT 跟踪。

可产出内容：

1. `runs/carrada_ra_gtbbox_sort_smoke/tracks/tracks.csv`。
2. 带 `track_id` 的跟踪可视化图。
3. 按 sequence 组织的跟踪帧序列。
4. MOTA 等基础跟踪指标。

展示价值：

1. 证明项目不止完成单帧目标框，还能把逐帧检测结果关联成轨迹。
2. 证明 `detections.csv -> tracks.csv -> visualization` 链路可运行。
3. 证明 SORT 参数可以通过 tracker config 控制，而不是硬编码在临时脚本中。

边界说明：

1. 当前 IDF1、ID switches、track fragmentation 被标记为 unavailable。
2. 这些 ID 类指标需要先审计 CARRADA `object_id` 在序列内是否稳定，再决定是否启用。

### 2.4 基础指标产物

可产出内容：

1. `metrics/detection_metrics.json`。
2. `metrics/tracking_metrics.json`。
3. `metrics/summary.csv`。
4. detection metrics：`precision`、`recall`、`F1`、`mAP50`。
5. tracking metrics：`MOTA`。
6. ID 类指标的 unavailable 状态和原因。

展示价值：

1. 证明项目具备机器可读的指标产物。
2. 证明检测和跟踪结果能在同一实验目录中被归档。
3. 证明当前不可用指标没有被静默伪造，而是明确记录状态和原因。

### 2.5 可视化与失败样例产物

可产出内容：

1. GT 可视化图。
2. detection 可视化图。
3. tracking 可视化图。
4. `visualizations/failures/failure_examples.json`。

展示价值：

1. 证明实验结果可以被人工复核。
2. 证明指标之外还能定位漏检、虚警或跟踪缺失等问题。
3. 证明项目输出不是一次性命令行结果，而是可归档、可查看的实验材料。

### 2.6 可复现实验归档

可产出内容：

1. `runs/carrada_ra_gtbbox_sort_smoke/config.yaml`。
2. `runs/carrada_ra_gtbbox_sort_smoke/detections/`。
3. `runs/carrada_ra_gtbbox_sort_smoke/tracks/`。
4. `runs/carrada_ra_gtbbox_sort_smoke/metrics/`。
5. `runs/carrada_ra_gtbbox_sort_smoke/visualizations/`。

展示价值：

1. 证明同一实验的配置、结果、指标和可视化可以集中归档。
2. 证明实验可以被复跑和审计。
3. 证明后续替换 YOLO 检测器时，不需要重写跟踪、评估和可视化模块。

## 3. 当前不能直接拿到的产物

### 3.1 真实 YOLO 推理产物

当前不能直接拿到真实 YOLO 推理产物。

阻塞原因：

1. 当前环境缺少 `torch`。
2. 当前环境缺少 `ultralytics`。
3. 当前缺少 `weights/yolo_pretrained.pt`。
4. 当前 `src/aerotrack/pipeline.py` 仅支持 `detector.source=gt_bbox`，YOLO inference adapter 尚未实现。

需要补齐后才能产出：

1. YOLO 推理检测图。
2. 由 YOLO 预测框转换得到的 `detections.csv`。
3. YOLO 检测指标。
4. YOLO 检测结果接 SORT 后的真实检测-跟踪 baseline。

### 3.2 真实 YOLO 训练产物

当前不能拿到真实 YOLO 训练产物。

阻塞原因：

1. 当前无可用 GPU。
2. 当前内存规模不适合完整 YOLO 训练。
3. 当前训练入口是 Stage1 保留入口，尚未实现真实训练执行。
4. 当前缺少训练权重和完整训练配置确认。

不能宣称的内容：

1. 已训练出 CARRADA Range-Angle YOLO 检测模型。
2. 已完成正式 YOLO 训练 baseline。
3. 当前检测性能代表训练后的雷达检测器水平。

### 3.3 ByteTrack 对比产物

当前不能拿到 ByteTrack 对比产物。

原因：

1. ByteTrack 在当前规格中属于后续接入或 v1.5 目标。
2. 当前跟踪模块只跑通 SORT baseline。
3. 当前没有 ByteTrack adapter 和同规约输出。

### 3.4 完整 ID 类跟踪指标

当前不能拿到完整 ID 类跟踪指标。

原因：

1. CARRADA 标注中的 `object_id` 稳定性尚未审计完成。
2. 当前实现将 IDF1、ID switches、track fragmentation 标记为 unavailable。
3. 在身份标注稳定性未确认前，不应输出看似有效的 ID 类指标。

## 4. CPU 环境下推荐执行顺序

### 4.1 复核环境和前置条件

执行目标：

1. 确认 Python 版本符合项目约束。
2. 确认 smoke 依赖可导入。
3. 确认 CARRADA 数据目录存在。
4. 确认 `gt_bbox` 实验配置可通过 preflight。

建议命令：

```bash
uv run aerotrack run-experiment --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml --preflight-only
```

预期结果：

1. `python` 检查为 OK。
2. `dependencies.smoke` 检查为 OK。
3. `dataset.root` 检查为 OK。
4. `detector.source` 显示为 `gt_bbox` smoke diagnostics。

### 4.2 重新生成 smoke 数据

执行目标：

1. 从 CARRADA 本地数据重新生成 smoke 数据集。
2. 更新 PNG、label、索引、标注和 GT 可视化检查图。

建议命令：

```bash
uv run aerotrack prepare-data --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
```

预期产物：

1. `data/processed/carrada_ra_smoke/sample_index.csv`。
2. `data/processed/carrada_ra_smoke/annotations.csv`。
3. `data/processed/carrada_ra_smoke/conversion_records.csv`。
4. `data/processed/carrada_ra_smoke/images/`。
5. `data/processed/carrada_ra_smoke/labels/`。
6. `data/processed/carrada_ra_smoke/visual_checks/gt/`。

### 4.3 运行完整 `gt_bbox + SORT` smoke 实验

执行目标：

1. 生成诊断检测结果。
2. 运行 SORT 跟踪。
3. 计算基础指标。
4. 输出可视化和失败样例。
5. 归档完整实验目录。

建议命令：

```bash
uv run aerotrack run-experiment --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
```

预期产物：

1. `runs/carrada_ra_gtbbox_sort_smoke/config.yaml`。
2. `runs/carrada_ra_gtbbox_sort_smoke/detections/detections.csv`。
3. `runs/carrada_ra_gtbbox_sort_smoke/tracks/tracks.csv`。
4. `runs/carrada_ra_gtbbox_sort_smoke/metrics/detection_metrics.json`。
5. `runs/carrada_ra_gtbbox_sort_smoke/metrics/tracking_metrics.json`。
6. `runs/carrada_ra_gtbbox_sort_smoke/metrics/summary.csv`。
7. `runs/carrada_ra_gtbbox_sort_smoke/visualizations/`。

### 4.4 单独复跑各阶段

执行目标：

1. 验证每个 stage 可以独立执行。
2. 方便定位失败发生在数据、检测、跟踪、评估还是可视化阶段。

建议命令：

```bash
uv run aerotrack run-detection --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
uv run aerotrack run-tracking --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
uv run aerotrack evaluate --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
uv run aerotrack visualize --config configs/experiment/carrada_ra_gtbbox_sort_smoke.yaml
```

预期结果：

1. 每个命令都能读取上游统一产物。
2. 每个阶段失败时能给出明确错误，而不是静默生成无效结果。

### 4.5 整理展示材料

执行目标：

1. 从已有产物中挑选可用于展示的图和表。
2. 明确区分诊断闭环和正式 YOLO baseline。

建议整理内容：

1. GT 标注图：从 `data/processed/carrada_ra_smoke/visual_checks/gt/` 挑选。
2. 检测图：从 `runs/carrada_ra_gtbbox_sort_smoke/visualizations/detections/` 挑选，并标注为 `gt_bbox` 诊断检测。
3. 跟踪图：从 `runs/carrada_ra_gtbbox_sort_smoke/visualizations/tracks/` 挑选。
4. 指标表：使用 `runs/carrada_ra_gtbbox_sort_smoke/metrics/summary.csv`。
5. 失败样例：使用 `runs/carrada_ra_gtbbox_sort_smoke/visualizations/failures/failure_examples.json`。

## 5. CPU 环境下的最小验收清单

完成以下内容即可认为 CPU 环境下的真实可产出材料已经闭环：

1. `prepare-data` 可以从本地 CARRADA 数据生成 processed smoke 数据。
2. `run-experiment` 可以完成 `gt_bbox + SORT` smoke 实验。
3. `detections.csv`、`tracks.csv`、`summary.csv` 均存在且字段符合项目规约。
4. `detection_metrics.json` 和 `tracking_metrics.json` 均存在。
5. `visualizations/gt`、`visualizations/detections`、`visualizations/tracks` 均有图像输出。
6. `failure_examples.json` 存在。
7. 文档或展示材料中明确写明 `gt_bbox` 是诊断来源，不是 YOLO baseline。

## 6. 后续接入 GPU 后的衔接点

CPU 阶段产物不会浪费。后续拿到 GPU 后，应该复用以下内容：

1. 复用 `data/processed/carrada_ra_smoke` 或扩展为更大规模 processed 数据。
2. 复用统一检测结果字段：`sequence_id, frame_id, class_id, score, x1, y1, x2, y2`。
3. 新增 YOLO inference adapter，将 YOLO 原始预测转换为统一 `detections.csv`。
4. 复用当前 SORT、评估和可视化模块。
5. 生成新的实验目录，例如 `runs/carrada_ra_yolo_sort_baseline/`。
6. 将 `gt_bbox + SORT` 作为诊断上限或链路对照，将 `YOLO + SORT` 作为真实检测-跟踪 baseline。
