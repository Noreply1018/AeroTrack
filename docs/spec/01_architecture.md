# 01. 系统架构

## 总体链路

v1 pipeline 固定为：

```text
CARRADA Range-Angle 数据
-> 数据准备
-> YOLO 检测训练/推理
-> 检测结果规约
-> SORT / ByteTrack 跟踪
-> 检测与跟踪评估
-> 可视化与轨迹回放
-> 实验归档
```

每个模块只依赖上游统一结果，不直接耦合具体实现细节。例如，跟踪模块只读取项目统一检测结果文件，不读取 YOLO 原始输出目录。

## 核心模块

### 数据准备

负责读取 CARRADA Range-Angle 数据、解析原始标注、生成检测框标注、划分数据集并输出 YOLO 可用目录结构。该模块还应提供样本级可视化检查，用于验证标注转换是否正确。

### 检测

负责启动 YOLO 训练或推理，并将 YOLO 原始输出转换为项目内部统一检测格式。检测模块应支持不同权重、输入尺寸、置信度阈值和 NMS 参数。

### 跟踪

负责读取统一检测结果，并按序列独立运行跟踪器。v1 默认跟踪器为 SORT，ByteTrack 使用相同输入输出规约预留。

### 评估

负责计算 detection metrics 和 tracking metrics，并输出机器可读指标文件和汇总表。评估模块应支持不同实验目录之间的横向对比。

### 可视化

负责生成检测框、跟踪 ID、轨迹回放和失败样例材料。可视化结果应从统一检测/跟踪结果生成，而不是依赖模型框架内部绘图。

### 实验编排

负责通过配置文件串联数据准备、检测、跟踪、评估和可视化。编排模块在运行前检查输入路径、权重路径和输出目录，在运行后保存配置副本、日志和结果摘要。

## 推荐工程结构

```text
configs/
  dataset/
  detector/
  tracker/
  experiment/
docs/
  spec/
src/
  data/
  detection/
  tracking/
  evaluation/
  visualization/
  pipeline/
scripts/
  prepare_data.py
  train_detector.py
  run_detection.py
  run_tracking.py
  evaluate.py
  visualize.py
  run_experiment.py
data/
runs/
weights/
```

`data/`、`runs/`、`weights/` 和模型权重文件必须进入 `.gitignore`。仓库只保留代码、配置模板、正式文档和必要的小规模示例。

默认数据根目录为 `data/carrada/`，默认权重根目录为 `weights/`。两者都必须能通过配置覆盖，不能硬编码在模块内部。

## 工程环境

v1 使用 `uv` 作为默认 Python 环境和依赖管理工具。仓库应通过 `pyproject.toml` 描述 Python 依赖，并提交 `uv.lock` 固化可复现环境。

Python 版本默认锁定为 3.11。PyTorch 的 CPU / GPU wheel 受本机 CUDA、驱动和平台影响，不在通用配置中盲目写死；若实验需要 GPU 版本，必须在环境说明或实验配置中明确记录安装来源和版本。

所有项目脚本默认通过 `uv run` 执行，例如：

```bash
uv run python scripts/prepare_data.py
uv run python scripts/run_detection.py
uv run python scripts/run_tracking.py
uv run python scripts/evaluate.py
```

`uv` 只负责 Python 依赖、虚拟环境和命令执行，不替代系统 CUDA 驱动、CARRADA 数据路径、模型权重路径或实验配置检查。

## 配置原则

一次实验应由一个 experiment config 描述。配置至少引用：

1. 数据集配置。
2. 检测器配置。
3. 跟踪器配置。
4. 评估配置。
5. 可视化配置。
6. 输出目录和实验名称。

运行时应把最终生效配置复制到实验输出目录，避免后续无法确认结果来源。
