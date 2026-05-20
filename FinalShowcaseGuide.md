# AeroTrack 最终展示材料

这份材料用于最终展示当前 CPU 环境下已经真实跑通的 AeroTrack 实验闭环。它的重点不是罗列产物，而是把“能展示什么、怎么解释、不能宣称什么”讲清楚。

当前展示口径是：AeroTrack 已经完成 CARRADA 雷达 Range-Angle 数据转换、统一检测结果生成、SORT 多目标跟踪、基础指标评估和可视化归档。检测来源是 `gt_bbox` 诊断检测，用来验证下游链路，不代表 YOLO 模型性能。

## 1. 总体展示图

![AeroTrack CPU 诊断闭环单帧大图](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_01_large_triptych.png)

这张图可以作为最终展示的主图。它只放一个典型帧，避免小图密集堆叠；左侧是全局 Range-Angle 雷达图，中间是目标区域放大图，右侧是同帧 camera reference。雷达图中的三色轮廓分别对应 GT 标注、由 `gt_bbox` 转换得到的诊断检测框和 SORT 输出的跟踪结果。

它能说明三件事：

1. CARRADA Range-Angle 雷达图已经可以叠加目标框进行人工复核。
2. 项目内部统一检测结果格式已经能被后续模块稳定消费。
3. 单帧检测结果已经接入 SORT，形成带 `track_id` 的跟踪输出。

展示时建议只围绕这张图讲“链路跑通”，不要把中间列解释成 YOLO 预测结果。

## 2. 连续帧跟踪展示

![AeroTrack SORT 连续帧跟踪展示](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_02_tracking_sequence.png)

这张图用于讲跟踪部分。它比密集总览图更适合答辩，因为画面保留连续帧 RA 跟踪裁剪，并为每帧配同帧 camera reference。观众可以直接观察 `track_id` 如何随时间出现在不同帧里，同时不会被抽象雷达图完全挡住理解。

这张图只说明 SORT 链路已经输出序列级轨迹。身份稳定性审计已经完成，但当前评估实现仍未启用 IDF1、ID switches 和 track fragmentation，因此这些指标继续保持 unavailable。

## 3. 指标表

| 展示项 | 结果 |
| --- | --- |
| 实验口径 | CPU 诊断闭环 |
| 检测来源 | `gt_bbox` |
| 评估 split | test |
| 评估帧数 | 264 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| mAP50 | 1.000 |
| MOTA | 0.520 |
| IDF1 / ID switches / track fragmentation | unavailable |

![AeroTrack CPU 诊断闭环指标页](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_03_metrics.png)

这张表适合放在成果页或实验结果页。解释时需要保持克制：

- 检测指标为 1.000 是诊断检测的预期表现，因为检测框来自 GT 标注转换。
- MOTA=0.520 说明跟踪评估链路已经产出机器可读结果，但它不是完整身份跟踪评估结论。
- ID 类指标暂不输出数值，因为当前评估实现尚未启用 IDF1、ID switches 和 track fragmentation 计算。
- 当前诊断报告口径下没有漏检或虚警样例；这只适用于 `gt_bbox` 诊断检测，不代表真实模型不会漏检或误检。

## 4. CPU 后续补强结果

在完成基础 smoke 闭环后，又继续补齐了四类 CPU 可做工作：身份稳定性审计、SORT 参数消融、展示大图增强和更大规模数据转换。

### 失败样例检查

![AeroTrack 失败样例检查页](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_04_failure_report.png)

这张图用于说明失败样例报告已经生成。当前 `gt_bbox` 诊断检测口径下，漏检帧和虚警帧数量为 0；ID switch 和 fragmentation 样例仍然保持 unavailable。展示时要强调这不是“真实模型没有失败”，而是“诊断检测链路没有发现检测层面的漏检或虚警”。

### 单目标细节图

![AeroTrack 单目标细节图](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_05_single_target_detail.png)

这张图用于单独解释一个目标在 Range-Angle 图中的位置。左侧保留全局雷达图，中间给出目标区域放大图，右侧补充同帧相机参考图，并用颜色区分 GT、诊断检测和 SORT track。这里的伪彩色和对比度裁剪只用于展示；坐标仍来自 CARRADA Range-Angle 标注，展示渲染优先读取原始 Range-Angle `.npy`，找不到时才回退到 prepared PNG。

### 连续轨迹条带图

![AeroTrack 连续轨迹条带图](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_06_track_strip.png)

这张图比四帧跟踪页更适合展示“同一个 ID 随时间延续”。它把同一 track ID 的多个连续 RA 裁剪和对应相机帧排成条带，适合在答辩中说明项目已经从单帧框推进到序列级轨迹复核。

### 身份稳定性审计

| 审计项 | 结果 |
| --- | --- |
| 标注行数 | 2,425 |
| 序列数 | 10 |
| 对象数 | 28 |
| class_id 变化对象数 | 0 |
| raw_label 变化对象数 | 0 |
| 帧间可见性 gap 对象数 | 0 |
| 审计结论 | pass |

这说明当前 10 序列扩展数据中的同一 `sequence_id/object_id` 没有发现类别或原始标签变化。它可以支撑“当前 CPU 扩展样本内身份标注没有发现明显不稳定”的说法，但还不能直接把完整 ID 类指标作为正式结论；还需要把 IDF1、ID switches 和 track fragmentation 的计算逻辑接入评估模块。

### SORT 参数消融

![AeroTrack SORT 参数消融图](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_07_sort_sweep.png)

| max_age | min_hits | IOU 阈值 | 轨迹数 | 轨迹行数 | MOTA | TP | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.10 | 732 | 2,425 | 0.700 | 363 | 0 | 0 |
| 1 | 1 | 0.30 | 1,232 | 2,425 | 0.468 | 363 | 0 | 0 |
| 3 | 1 | 0.30 | 1,105 | 2,425 | 0.463 | 363 | 0 | 0 |
| 5 | 1 | 0.30 | 1,085 | 2,425 | 0.460 | 363 | 0 | 0 |
| 3 | 2 | 0.30 | 481 | 1,320 | 0.292 | 181 | 0 | 182 |
| 3 | 1 | 0.50 | 1,607 | 2,425 | 0.259 | 363 | 0 | 0 |

这张表可以说明 SORT 参数会明显影响轨迹连续性和 MOTA。当前诊断检测没有漏检或虚警，所以不同参数的差异主要来自轨迹关联策略，而不是检测模型能力。

### 10 序列 CPU 扩展数据

![AeroTrack CPU 规模扩展对比图](runs/carrada_ra_gtbbox_sort_smoke/showcase/slides/slide_08_scale_comparison.png)

| 展示项 | smoke 闭环 | 10 序列扩展 |
| --- | ---: | ---: |
| 样本帧数 | 1,281 | 5,727 |
| 标注数 | 436 | 2,425 |
| 检测结果数 | 436 | 2,425 |
| 跟踪结果数 | 436 | 2,425 |
| 评估 test 帧数 | 264 | 528 |
| Precision | 1.000 | 1.000 |
| Recall | 1.000 | 1.000 |
| F1 | 1.000 | 1.000 |
| mAP50 | 1.000 | 1.000 |
| MOTA | 0.520 | 0.463 |

这张表用于证明 CPU 上不只跑了最小 smoke 样例，还可以扩大到更多 CARRADA 序列进行转换和闭环评估。10 序列扩展仍然使用 `gt_bbox` 诊断检测，因此检测指标不能解释为 YOLO 模型效果。

## 5. 展示说明口径

最终汇报中可以直接使用这段说明：

> 当前 CPU 环境已经完成 CARRADA Range-Angle 数据转换、统一检测结果生成、SORT 多目标跟踪、基础指标评估和可视化归档。为了避免把环境诊断包装成模型效果，本阶段使用 `gt_bbox` 作为诊断检测来源，验证检测到跟踪的工程链路。正式 YOLO baseline 需要在补齐推理依赖、模型权重和 YOLO adapter 后重新产出。

如果需要一句话版本，可以使用：

> 本阶段证明 AeroTrack 的雷达检测-跟踪实验闭环已经跑通；当前检测来源是 `gt_bbox` 诊断输入，不代表 YOLO 模型性能。

## 6. 可以宣称的结论

当前材料可以支持以下结论：

- CARRADA Range-Angle 数据已经转换成检测任务可用的数据形态。
- 项目已经跑通从检测结果到 SORT 跟踪结果的完整下游链路。
- 检测指标、跟踪指标、失败样例和可视化可以统一归档并被复核。
- 当前 10 序列扩展数据身份稳定性审计没有发现 class_id 或 raw_label 变化。
- SORT 参数消融已经说明跟踪结果对关联阈值和确认策略敏感。
- CPU 环境已经扩展到 10 个 CARRADA 序列的转换和诊断闭环。
- 后续接入 YOLO 检测器时，可以复用当前跟踪、评估和可视化模块。

当前材料不能支持以下说法：

- 已经训练出 CARRADA Range-Angle YOLO 检测模型。
- 当前 Precision、Recall、F1 或 mAP50 代表 YOLO 模型真实性能。
- 已经完成 ByteTrack 对比实验。
- 已经完成完整 ID 类跟踪指标评估。

## 7. 答辩展示建议

建议最终展示控制在八页：

1. 第一页放总体展示图，讲清楚“GT、诊断检测、SORT 跟踪”的关系。
2. 第二页放连续帧跟踪图，强调项目不止单帧检测，还输出序列级轨迹。
3. 第三页放单目标细节图，解释伪彩色 RA 图和目标框。
4. 第四页放轨迹条带图，展示同一 track ID 的连续帧。
5. 第五页放指标表，说明 CPU 诊断闭环已经产出检测与跟踪指标。
6. 第六页放失败样例检查页，说明当前诊断报告状态。
7. 第七页放 SORT 参数消融图，说明跟踪参数影响。
8. 第八页放 10 序列扩展图和边界说明，明确 `gt_bbox` 是链路诊断，不是 YOLO baseline。
