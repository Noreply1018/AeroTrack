# 08. 路线图

## v1: CARRADA Range-Angle Baseline

目标是建立稳定、可复现的最小工程闭环：

1. CARRADA Range-Angle 数据准备。
2. YOLO 数据集生成。
3. YOLO 推理和统一检测结果导出。
4. SORT 跟踪和统一跟踪结果导出。
5. detection / tracking metrics。
6. 检测可视化、跟踪可视化和轨迹回放。
7. 实验配置、日志和输出归档。

v1 的重点是把接口和流程做稳，不追求覆盖所有模型和表示。

## v1.5: ByteTrack 与实验对比

在 v1 稳定后接入 ByteTrack：

1. 使用与 SORT 相同的输入输出规约。
2. 支持 SORT / ByteTrack 横向比较。
3. 补充 ID switch、track fragmentation 等失败案例可视化。
4. 形成第一版 ablation 表。

## v2: 多表示与检测器扩展

扩展输入和检测器：

1. 增加 Range-Doppler、Doppler-Angle 或多表示融合输入。
2. 抽象 detector adapter，支持更多检测模型。
3. 在同一评估口径下比较不同表示和模型。
4. 完善数据转换记录和表示级配置。

## v3: 系统接口与硬件承接

在实验基线成熟后，为真实系统承接做接口收束：

1. 固化在线输入接口。
2. 固化检测和跟踪输出协议。
3. 评估推理速度和资源占用。
4. 形成可接入硬件或实时数据流的系统接口方案。

## 长期方向

长期可建设为雷达感知实验平台：

1. 实验注册表和结果查询。
2. 自动生成对比报告。
3. 失败样例库。
4. 多数据集 benchmark。
5. 与真实雷达系统对接。
