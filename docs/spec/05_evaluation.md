# 05. 评估规格

## 评估目标

评估模块负责在统一口径下计算检测和跟踪指标，使不同检测模型、跟踪器和实验配置可以被复核和横向比较。

## 检测指标

v1 至少输出以下 detection metrics：

1. precision。
2. recall。
3. F1。
4. mAP50。

指标计算应基于统一检测结果和内部标注格式。IOU 阈值、类别过滤和 split 必须写入评估配置。

mAP50 可先使用项目内轻量实现，但匹配逻辑必须透明、可测试，并明确使用按类别匹配和一对一贪心匹配或等价策略。若后续切换到 pycocotools 等外部实现，必须在指标文件中记录实现名称和版本，避免不同实现的结果直接混比。

## 跟踪指标

v1 至少输出以下 tracking metrics：

1. MOTA。
2. IDF1。
3. ID switches。
4. track fragmentation。

若数据标注无法稳定支持某些 ID 类指标，评估模块必须在结果中标记该指标不可用，并说明原因，不能静默输出无意义数值。

指标 JSON 中不可用指标统一表示为：

```json
{
  "value": null,
  "status": "unavailable",
  "reason": "ground-truth object_id is not stable within sequence"
}
```

跟踪指标默认使用 `motmetrics` 计算。IOU 匹配阈值、类别过滤、split、是否按类别评估以及 `motmetrics` 版本必须进入评估配置或指标元信息。

## 评估输入

评估输入包括：

1. 内部标注文件。
2. 统一检测结果文件。
3. 统一跟踪结果文件。
4. 类别映射。
5. 评估配置。

评估模块不得直接读取 YOLO、SORT 或 ByteTrack 的框架私有输出。

## 评估输出

每次评估至少输出：

```text
metrics/
  detection_metrics.json
  tracking_metrics.json
  summary.csv
```

JSON 文件用于机器读取，CSV 文件用于实验横向对比和汇报。

## 对比口径

不同实验之间的对比必须保证：

1. 使用相同 split。
2. 使用相同类别映射。
3. 使用相同评估阈值。
4. 明确检测器、跟踪器和参数差异。

不满足这些条件的结果不能放入同一 baseline 对比表。

## 失败样例关联

评估结果应为可视化模块提供失败样例线索，例如：

1. 漏检帧。
2. 虚警帧。
3. ID switch 发生帧。
4. 轨迹断裂片段。

这些线索用于生成失败样例导出，不影响指标本身。
