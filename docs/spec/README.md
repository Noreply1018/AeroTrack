# 雷达检测-跟踪实验基线规格文档

本目录存放项目的正式工程规格文档。文档目标是约束后续实现，而不是记录项目流程材料或本地草稿。

## 阅读顺序

1. [00_overview.md](00_overview.md)：项目定位、v1 目标、边界与验收摘要。
2. [01_architecture.md](01_architecture.md)：系统架构、模块关系与推荐工程结构。
3. [02_data_contract.md](02_data_contract.md)：CARRADA Range-Angle 数据、标注转换与数据接口。
4. [03_detection.md](03_detection.md)：YOLO 检测训练、推理和检测结果规约。
5. [04_tracking.md](04_tracking.md)：SORT / ByteTrack 跟踪接口与结果规约。
6. [05_evaluation.md](05_evaluation.md)：检测和跟踪指标、实验对比口径。
7. [06_visualization.md](06_visualization.md)：检测可视化、轨迹回放和失败样例导出。
8. [07_experiments.md](07_experiments.md)：实验目录、baseline、ablation 和复现规则。
9. [08_roadmap.md](08_roadmap.md)：v1 到后续版本的演进路线。

## v1 核心承诺

v1 固定以 CARRADA 数据集的 Range-Angle 表示作为主要输入，构建可复现的检测-跟踪实验基线：

```text
CARRADA Range-Angle 数据
-> 标注转换与样本组织
-> YOLO 检测训练/推理
-> SORT 默认跟踪，ByteTrack 同接口预留
-> detection / tracking metrics
-> 结果可视化、轨迹回放与实验归档
```

## 文档维护原则

1. 规格文档必须面向实现和验收，不记录队伍与流程类信息。
2. 接口规约优先写在本目录，而不是散落在代码注释或临时脚本里。
3. 新增模块前先补齐对应规格，避免实现先行导致接口口径不一致。
4. 数据、权重、实验输出不进入 git；代码、配置模板和正式规格文档进入 git。
