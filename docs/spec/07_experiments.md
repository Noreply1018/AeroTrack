# 07. 实验规格

## 实验目标

实验规格用于保证每次运行都能被复现、比较和归档。任何 baseline 或 ablation 都应通过配置描述，而不是依赖临时命令或手工修改代码。

## 实验目录

每次实验应生成独立目录：

```text
runs/<experiment_name>/
  config.yaml
  detections/
  tracks/
  metrics/
  visualizations/
  logs/
```

`config.yaml` 是最终生效配置的副本。若运行时合并了多个配置文件，保存到实验目录中的配置必须是合并后的完整配置。

## Baseline

v1 baseline 固定为：

1. 数据集：CARRADA。
2. 表示：Range-Angle。
3. 检测器：YOLO。
4. 跟踪器：SORT。
5. 评估：detection metrics + tracking metrics。
6. 输出：统一结果文件 + 可视化 + 实验目录。

该 baseline 是后续所有 ablation 的对照组。

v1 第一阶段先跑通 `数据准备 -> 检测推理 -> SORT 跟踪 -> 评估 -> 可视化 -> 实验归档` 的最小闭环。YOLO 训练脚本和配置必须保留在工程中，但不阻塞第一阶段闭环验收；正式 baseline 对外发布前应补齐训练过程、训练配置和训练产物记录。

## Ablation

baseline 跑通后，按优先级开展以下对比：

1. 不同 YOLO 权重或模型规模。
2. 不同置信度阈值和 NMS 设置。
3. SORT 与 ByteTrack 的跟踪结果对比。
4. 不同雷达表示的预留对比，例如 Range-Doppler 或 RA/RD 融合。

v1 不要求完成所有 ablation，但工程接口必须支持新增实验时不破坏现有流程。

## 实验命名

实验名应包含核心变量，推荐格式：

```text
<dataset>_<representation>_<detector>_<tracker>_<tag>
```

示例：

```text
carrada_ra_yolo_sort_baseline
```

## 复现规则

一次实验可复现需要满足：

1. 实验目录保存完整配置。
2. 指标文件和可视化结果来自同一组检测/跟踪输出。
3. 日志记录关键命令、开始时间、结束时间和异常。
4. 若使用外部权重，配置中必须记录权重路径或权重标识。
5. 若数据划分不是官方划分，实验目录必须能追溯划分文件。
6. 运行命令默认使用 `uv run`，依赖版本由 `pyproject.toml` 和 `uv.lock` 固化。
7. 若使用 GPU 版 PyTorch 或特殊 CUDA wheel，实验日志或环境说明必须记录安装来源、CUDA 版本和 PyTorch 版本。

## 横向比较

实验比较表只允许纳入可比实验。可比实验必须使用相同数据 split、类别映射和评估阈值。若这些条件不同，应在表中分组展示，不能直接比较数值。
