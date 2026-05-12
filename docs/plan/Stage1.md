# AeroTrack 第一阶段实施计划

本文档沉淀 AeroTrack v1 第一阶段计划。第一阶段目标不是训练模型或追求检测精度，而是在本机环境约束下尽快跑通可复现的最小工程闭环，并优先产出可检查的可视化图。

## 1. 阶段目标

第一阶段固定采用 CARRADA 数据集的 Range-Angle 表示，先使用小样本数据完成以下链路：

```text
CARRADA 数据获取
-> Range-Angle 数据准备
-> PNG 图像与样本索引生成
-> GT 标注可视化
-> YOLO 预训练权重推理可视化
-> 统一 detections.csv
-> SORT 跟踪
-> 统一 tracks.csv
-> 基础指标与实验归档
```

第一阶段的核心验收是“能稳定出图、能复现流程、接口格式正确”。YOLO 训练、ByteTrack、完整 ablation 和多雷达表示不进入第一阶段阻塞项。

第一阶段验收拆成两个层级：

1. **smoke 诊断闭环**：允许使用 `gt_bbox` 作为检测来源，用于验证数据契约、SORT、评估、可视化和实验归档。
2. **YOLO 推理闭环**：使用 Ultralytics YOLO 预训练或外部权重完成推理并导出统一 `detections.csv`。该层级对应正式检测 baseline 的第一步；若因环境或权重问题跳过，必须记录为“YOLO 推理闭环未完成”，不能把 `gt_bbox` 结果视为 YOLO baseline。

## 2. 当前前提

本机适合先做推理和工程闭环验证：

1. CPU 资源充足，适合数据转换、可视化和小规模 CPU 推理。
2. WSL 当前内存约 8GB，完整 YOLO 训练不理想。
3. 当前 Linux 侧尚未确认可用 `nvidia-smi` 和 CUDA PyTorch。
4. `uv` 已安装，后续工程环境按规格锁定 Python 3.11。
5. 本地尚无 CARRADA 数据，需要先下载或准备到 `data/carrada/`。

因此第一阶段采用“不训练、先小样本、先出图”的策略。

## 3. 数据获取计划

数据根目录默认使用：

```text
data/carrada/
```

CARRADA 不进入 git。后续实现前先探测官方下载入口、文件结构和文件大小，再决定下载方式。当前计划使用公开入口：

```text
https://arthurouaknine.github.io/codeanddata/carrada
http://download.tsi.telecom-paristech.fr/Carrada
```

数据获取步骤：

1. 确认远端目录、压缩包名称、文件大小和可访问性。
2. 下载到 `data/carrada/` 或临时下载目录后解压。
3. 保留原始数据目录结构，不在数据目录中混入实验输出。
4. 在配置中记录数据根路径，禁止在代码中硬编码本机绝对路径。
5. 下载前检查磁盘空间。CARRADA 主数据压缩包和解压目录体积较大，第一阶段不得默认下载额外的大体积 RAD tensor 包，除非后续实验明确需要。
6. 第一阶段只抽取少量 sequence 做 smoke 实验，避免一开始全量处理拖慢验证。

## 4. 第一批可视化目标

第一批图按优先级产出：

1. **GT 标注可视化图**：基于 CARRADA 标注生成 bbox，并叠加到 Range-Angle PNG 上。该图用于验证数据转换、坐标转换、类别映射和标注读取是否正确。
2. **YOLO 预训练权重推理图**：使用外部通用 YOLO 权重对 Range-Angle PNG 推理，并绘制预测框。该图只验证检测 adapter、推理调用和统一检测格式，不作为雷达检测效果判断。
3. **SORT 跟踪可视化图**：基于统一检测结果生成跟踪 ID 图，用于验证 `detections.csv -> tracks.csv -> visualization` 链路。

通用 YOLO 权重并非雷达 Range-Angle 数据训练所得，预测框可能不准确。第一阶段应以 GT 标注可视化作为可信检查图，YOLO 推理图作为工程链路验证图。

## 5. 工程里程碑

### 5.1 工程骨架

建立推荐目录结构：

```text
configs/
  dataset/
  detector/
  tracker/
  experiment/
src/
  data/
  detection/
  tracking/
  evaluation/
  visualization/
  pipeline/
scripts/
```

同时补齐：

1. `pyproject.toml`。
2. Python 3.11 约束。
3. `uv.lock`。
4. 基础依赖配置。
5. 命令行脚本入口。
6. YOLO 训练脚本入口和训练配置模板。

第一阶段保留 `scripts/train_detector.py` 和 detector train config 的工程入口，但不执行 YOLO 训练，也不把训练成功作为验收条件。训练入口应在依赖、数据或权重不满足时给出明确错误，不能静默生成无效结果。

### 5.2 配置与实验目录

实现 experiment config，统一描述数据、检测、跟踪、评估、可视化和输出目录。

每次实验输出：

```text
runs/<experiment_name>/
  config.yaml
  detections/
  tracks/
  metrics/
  visualizations/
  logs/
```

运行时保存最终生效配置副本，避免后续无法复现实验来源。

### 5.3 数据准备最小版

实现 CARRADA Range-Angle 小样本转换：

1. 读取 CARRADA 本地根目录。
2. 解析少量 sequence 的 Range-Angle 数据和标注。
3. 将 Range-Angle 数值按配置归一化为 8-bit 三通道 PNG。
4. 生成 `sample_index.csv`。
5. 生成内部标注 CSV。
6. 生成 YOLO 可读取数据集目录。
7. 生成 YOLO 标签文件。
8. 生成 train / val / test 划分文件。
9. 生成类别定义文件。
10. 生成原始标注到检测框标注的转换记录。
11. 输出抽样 GT 标注可视化检查图。

`sample_index.csv` 字段沿用规格：

```text
sample_id, sequence_id, frame_id, split, representation, image_path, label_path
```

内部标注字段沿用规格：

```text
sequence_id, frame_id, object_id, class_id, x1, y1, x2, y2
```

数据转换必须记录：

1. Range-Angle 归一化规则，并随 experiment config 归档。
2. 图像生成过程中的 resize、padding、crop 参数；如发生几何变换，必须同步转换标注坐标。
3. 标注来源类型，例如 `mask_bbox`、`point_expand` 或 `sparse_expand`。
4. 扩框参数、裁剪状态，以及空标注帧是保留还是过滤。
5. 类别映射和 `class_id` 顺序，且一次实验内不可临时重编号。
6. 数据划分来源；若不用官方划分，自定义划分必须按 sequence 级别执行，并记录随机种子和生成规则。

若 `object_id` 稳定性暂时无法确认，先保留字段并在跟踪 ID 类指标中按统一 JSON 结构标记不可用。

### 5.4 检测结果接口

第一阶段支持两种检测来源：

1. `gt_bbox`：把 GT bbox 转成带高置信度的 `detections.csv`，用于稳定验证后续跟踪和可视化。
2. `yolo_pretrained`：调用 Ultralytics YOLO 预训练权重推理，转换为项目统一检测格式。

`gt_bbox` 是诊断和控制链路，只用于验证统一结果、SORT、评估和可视化是否可运行，不能视为完成 YOLO 检测 baseline。正式检测链路必须通过 detector adapter 读取 Ultralytics 推理输出，并转换为统一 `detections.csv`。

一次 experiment 只激活一个 detection source，避免不同来源覆盖同一个 `detections.csv` 或混淆指标来源。若需要同时比较 `gt_bbox` 和 `yolo_pretrained`，必须生成两个独立实验目录，例如：

```text
runs/carrada_ra_gtbbox_sort_smoke/
runs/carrada_ra_yolopretrained_sort_smoke/
```

每个实验的 `config.yaml`、`metrics/*.json`、`metrics/summary.csv` 和可视化目录都必须记录当前使用的 detection source。

YOLO 预训练推理在第一阶段尽量执行；若因网络、权重下载、PyTorch 安装、系统资源或环境问题跳过，实验日志和 `summary.csv` 必须记录跳过原因。跳过 YOLO 预训练推理不阻塞 `gt_bbox` smoke 诊断闭环，但会导致 YOLO 推理闭环未完成，正式检测 baseline 也不能标记为完成。

统一检测结果字段：

```text
sequence_id, frame_id, class_id, score, x1, y1, x2, y2
```

默认输出：

```text
runs/<experiment_name>/detections/detections.csv
```

### 5.5 SORT 跟踪

实现 SORT baseline，输入统一检测结果，输出统一跟踪结果。

默认策略：

1. 按 `sequence_id` 分组独立跟踪。
2. 在每个 sequence 内默认再按 `class_id` 分组跟踪。
3. 跟踪参数来自 tracker config，不硬编码在脚本中。

统一跟踪结果字段：

```text
sequence_id, frame_id, track_id, class_id, score, x1, y1, x2, y2
```

默认输出：

```text
runs/<experiment_name>/tracks/tracks.csv
```

### 5.6 基础评估

第一阶段评估以接口正确和可复现为主：

1. 检测指标输出 precision、recall、F1、mAP50。
2. mAP50 使用项目内轻量实现时，必须明确按类别匹配和一对一贪心匹配逻辑，并为核心匹配函数补测试。
3. 跟踪指标输出 MOTA、IDF1、ID switches、track fragmentation。
4. 若 GT `object_id` 稳定性无法确认，IDF1、ID switches、track fragmentation 必须按统一 JSON 结构输出不可用状态和原因。
5. 跟踪指标默认使用 `motmetrics`；指标文件必须记录 `motmetrics` 版本、split、IOU 阈值、类别映射、是否按类别评估和实现名称。

不可用指标统一表示为：

```json
{
  "value": null,
  "status": "unavailable",
  "reason": "ground-truth object_id is not stable within sequence"
}
```

输出：

```text
runs/<experiment_name>/metrics/detection_metrics.json
runs/<experiment_name>/metrics/tracking_metrics.json
runs/<experiment_name>/metrics/summary.csv
```

### 5.7 可视化

可视化只读取统一产物，不读取 YOLO 或 SORT 私有输出。

第一阶段输出：

1. GT 标注检查图。
2. 检测框图。
3. 跟踪 ID 图。
4. 按 sequence 组织的帧目录。
5. 基础轨迹回放帧序列。
6. 基础失败样例清单。

第一阶段的轨迹回放先输出帧序列，不强制生成视频文件；视频导出作为增强项。失败样例导出先覆盖评估能稳定产生的漏检、虚警线索，并关联 `sequence_id` 和 `frame_id`；ID switch 和 track fragmentation 的失败样例在 ID 类指标可用后补齐。若 ID 类指标不可用，失败样例清单中也必须记录对应项的 `skipped` 或 `unavailable` 原因。

建议目录：

```text
runs/<experiment_name>/visualizations/
  gt/
  detections/
  tracks/
  sequences/
  failures/
```

文件名必须包含 `sequence_id` 和 `frame_id`，便于回溯原始样本。

### 5.8 一键实验脚本

提供统一入口：

```bash
uv run python scripts/run_experiment.py --config configs/experiment/carrada_ra_sort_smoke.yaml
```

第一阶段 smoke 实验默认执行：

```text
检查配置
-> 准备小样本数据
-> 生成 GT 可视化
-> 生成 gt_bbox detections.csv
-> 运行 SORT
-> 输出基础指标
-> 输出检测/跟踪可视化
-> 输出基础失败样例清单
-> 归档配置与日志
```

YOLO 预训练推理通过单独 experiment config 执行。若跳过，必须在日志、最终配置或 `summary.csv` 中记录原因。

## 6. 验收标准

smoke 诊断闭环通过条件：

1. CARRADA 小样本数据能从本地 `data/carrada/` 读取。
2. 能生成 Range-Angle PNG、YOLO 可读取数据集目录、`sample_index.csv`、内部标注 CSV、类别定义文件、划分文件和标注转换记录。
3. `sample_index.csv`、内部标注 CSV、`detections.csv` 和 `tracks.csv` 字段符合 `docs/spec` 规约。
4. 能输出至少一批 GT 标注可视化图。
5. 能通过 `gt_bbox` 生成统一 `detections.csv`，并明确标记该结果只用于诊断链路。
6. 能运行 SORT 并生成统一 `tracks.csv`。
7. 能生成检测框图、跟踪 ID 图和基础轨迹回放帧序列。
8. 能输出 detection metrics、tracking metrics、`summary.csv` 和基础失败样例清单。
9. ID 类指标不可用时，能按统一 JSON 结构输出 `unavailable` 和原因。
10. 能通过一个 experiment config 复现完整 smoke 流程。
11. 数据、权重和实验输出不进入 git。

YOLO 推理闭环通过条件：

1. 能使用 YOLO 预训练权重生成推理图。
2. 能把 YOLO 推理结果转换为统一 `detections.csv`。
3. 能基于 YOLO 推理结果继续跑 SORT、评估和可视化。
4. YOLO 推理实验的 `config.yaml`、日志、指标和 `summary.csv` 明确记录权重来源、推理参数和 detection source。

增强验收：

1. 能导出轨迹回放视频文件。

## 7. 暂不纳入第一阶段

以下内容延后：

1. YOLO 训练执行与训练产物验收。
2. ByteTrack 接入。
3. 完整 CARRADA 全量实验。
4. 多雷达表示，例如 Range-Doppler 或 Doppler-Angle。
5. 大规模 ablation。
6. 自动化报告系统。
7. GPU/CUDA 环境专项优化。
8. 轨迹回放视频强制生成。
9. ID switch 和 track fragmentation 失败样例完整导出。

## 8. 风险与处理策略

### 8.1 CARRADA 下载或结构变化

风险：公开下载入口、压缩包名称或目录结构可能变化。

处理：

1. 先探测远端和本地结构，再写解析逻辑。
2. 数据路径全部配置化。
3. 解析失败时输出明确错误，提示期望目录和实际目录。

### 8.2 标注格式理解偏差

风险：CARRADA 标注、mask、point 或 object ID 的语义可能需要结合真实文件确认。

处理：

1. 先用少量 sequence 做可视化人工检查。
2. 数据转换记录保留标注来源类型。
3. 对不稳定 `object_id` 不强行计算 ID 类指标。

### 8.3 YOLO 预训练结果不可靠

风险：通用视觉 YOLO 权重对雷达 Range-Angle 图没有语义适配。

处理：

1. 不把 YOLO 预训练图作为效果验收。
2. 使用 GT 标注可视化验证数据和坐标。
3. 使用 YOLO 预训练图仅验证推理 adapter 和统一输出格式。

### 8.4 本机资源不足

风险：WSL 内存偏小，GPU/CUDA 未就绪。

处理：

1. 第一阶段仅跑小样本。
2. 不训练 YOLO。
3. 默认支持 CPU 推理。
4. 后续如需训练，再单独规划 CUDA/PyTorch 和 WSL 内存配置。

### 8.5 磁盘空间不足

风险：CARRADA 主数据压缩包和解压目录体积较大，若后续下载额外 RAD tensor 包，磁盘占用会进一步增加。

处理：

1. 下载前记录远端文件大小并检查本地剩余空间。
2. 优先下载最小必要包；若官方包不能按表示或小样本粒度拆分，则记录原因和体积后再执行。
3. 临时压缩包、解压目录和实验输出必须放在 git 忽略路径下。
4. 不默认下载第一阶段不需要的大体积扩展数据。

## 9. 建议执行顺序

1. 建立工程骨架和依赖配置。
2. 实现配置读取和实验目录创建。
3. 探测并准备 CARRADA 小样本数据。
4. 实现 Range-Angle PNG 转换和样本索引。
5. 实现类别定义、划分文件、YOLO 数据集目录和标注转换记录。
6. 实现 GT bbox 转换和 GT 可视化。
7. 实现 `gt_bbox -> detections.csv` 诊断链路。
8. 实现 SORT 跟踪和 `tracks.csv`。
9. 实现基础指标输出和失败样例线索。
10. 实现检测/跟踪可视化和轨迹回放帧序列。
11. 接入 Ultralytics YOLO 预训练推理作为增强链路。
12. 用 smoke config 全流程验证。
13. 审计、修订；如用户要求提交，或项目约定需要提交，则只提交本任务相关文件。

## 10. 第一阶段完成后的下一步

第一阶段完成后，再进入 v1 后半段：

1. 扩大小样本到官方 split 或完整 test split。
2. 补齐正式训练配置、训练日志归档和训练产物记录。
3. 确认或引入雷达数据训练权重。
4. 完善 tracking metrics 的 ID 类指标。
5. 接入 ByteTrack 做 v1.5 对比。
