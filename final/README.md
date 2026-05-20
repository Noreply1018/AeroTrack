# AeroTrack 结项展示包

本目录汇总了本项目从数据转换、检测训练、SORT 跟踪到展示报告的结项材料。内容面向 PPT 制作，图片和表格已经按主题归档。

## 目录

- `figures/data_conversion/`：CARRADA Range-Angle 转换后的 GT 标注抽查图。
- `figures/yolo_training/`：YOLO 训练曲线、PR/F1 曲线、混淆矩阵、验证批次图。
- `figures/yolo_predictions/`：YOLO 推理展示图，当前复制 `18` 张。
- `figures/diagnostic_pipeline/`：GT 诊断检测 + SORT 跟踪闭环的大图。
- `figures/tracking/`：SORT 连续帧跟踪可视化。
- `tables/`：训练指标、诊断实验指标、数据规模和归档清单。
- `reports/`：中文展示报告和 YOLO 训练说明。
- `commands/`：复现实验命令。

## 当前真实结果口径

- 数据集：`carrada_ra_cpu10`，共 `5727` 张 RA PNG，`2425` 个标注框。
- YOLO 训练目录：`runs/yolo_final_demo/carrada_ra_cpu10_yolov8n_e30_cpu`。
- YOLO 预测目录：`runs/yolo_final_demo/carrada_ra_cpu10_showcase_pred`。
- YOLO 当前已记录 epoch：`30`。
- YOLO 当前 mAP50：`0.0157`。
- 诊断闭环实验数量：`3`。

## 展示边界

`YOLO` 章节来自真实 Ultralytics 训练输出；`diagnostic_pipeline` 章节来自 `gt_bbox` 诊断检测，用来展示下游 SORT、评估和可视化闭环。两者不能混写成同一个指标结论。
