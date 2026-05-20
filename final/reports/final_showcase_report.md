# 最终展示报告

## 展示主线

本展示包建议围绕“数据转换 -> YOLO 检测训练 -> SORT 跟踪闭环 -> 指标与可视化归档”展开。`figures/diagnostic_pipeline/slide_01_large_triptych.png` 适合作为主图，用一页解释 RA 图、目标框和轨迹输出之间的关系。

## 数据转换成果

`carrada_ra_cpu10` 包含 `5727` 张 RA PNG，标注框 `2425` 个。`figures/data_conversion/` 中的 GT 可视化图可用于说明标注框已经正确叠加在 RA 图上。

## 诊断闭环指标

| 实验 | sequence 数 | test 帧数 | precision | recall | mAP50 | MOTA | IDF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| carrada_ra_gtbbox_sort_smoke | 2 | 264 | 1.000 | 1.000 | 1.000 | 0.520 | unavailable |
| carrada_ra_gtbbox_sort_cpu10 | 10 | 528 | 1.000 | 1.000 | 1.000 | 0.463 | unavailable |
| carrada_ra_gtbbox_sort_server30 | 30 | 2130 | 1.000 | 1.000 | 1.000 | 0.412 | unavailable |

这些指标来自 `gt_bbox` 诊断检测，检测框由 GT 标注转换得到，因此检测指标接近理想值是预期现象。该章节用于证明后处理链路可运行，不能作为 YOLO 精度结论。

## SORT 参数消融

| max_age | min_hits | IOU 阈值 | 轨迹数 | MOTA | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.10 | 732 | 0.700 | 363 | 0 | 0 |
| 1 | 1 | 0.30 | 1232 | 0.468 | 363 | 0 | 0 |
| 3 | 1 | 0.30 | 1105 | 0.463 | 363 | 0 | 0 |
| 5 | 1 | 0.30 | 1085 | 0.460 | 363 | 0 | 0 |
| 3 | 2 | 0.30 | 481 | 0.292 | 181 | 0 | 182 |
| 3 | 1 | 0.50 | 1607 | 0.259 | 363 | 0 | 0 |

SORT 消融可用于说明跟踪结果受关联阈值和轨迹确认策略影响。当前 IDF1、ID switches 和 fragmentation 仍未接入正式评估，应在答辩中保持 unavailable 口径。

## 推荐图片

- `figures/yolo_training/results.png`：YOLO 训练曲线总览。
- `figures/yolo_training/BoxPR_curve.png`：PR 曲线。
- `figures/yolo_predictions/`：YOLO 推理展示图。
- `figures/diagnostic_pipeline/slide_02_tracking_sequence.png`：连续帧 SORT 跟踪。
- `figures/diagnostic_pipeline/slide_08_scale_comparison.png`：smoke 与 cpu10 规模对比。
