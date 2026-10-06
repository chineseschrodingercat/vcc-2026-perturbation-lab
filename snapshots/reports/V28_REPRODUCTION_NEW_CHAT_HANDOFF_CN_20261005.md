# V28 复现与新对话交接

用户最新要求：只完成 V28 复现，然后停止本会话的训练、测试、下载和提交。下一对话先读取本交接并等待用户指示；不要沿用此前持续运行许可自动启动实验或目标。

已依用户要求建立本项目新对话“V28 基线交接”，threadId `01a10b32-021f-7d51-8cf3-f981e1bf3e11`；初始指示仅阅读交接并等待，不启动任何实验或目标。

## 复现状态

- 原正式提交包、模型、特征、A/B/C 总效应、NB θ 和 donor 文件均保留，已实际重新核验文件哈希。
- 原完整包：`C:/00 - Competition/archive/legacy_20261003/runs/submission_v28_20261001/v28_n1_prediction.vcc`，3,145,994,240 B，SHA256 `f82fb2d796965026fb6ba84e22545c1af0bf78e512a13b7c3bd0f15f2c7c4523`；与原 submission_started.json 一致。原独立 H5AD 已清理，完整包仍在。
- **2026-10-05 08:32 UTC 已实际完成并停止：** 原正式 V28 的 A/B/C 900 组均值重放 PASS。每个背景的总效应和残差均为 float32 逐位 100% 一致，最大绝对误差及 RMS 都为 0；新三个压缩 NPZ 的 SHA256 也与原文件完全相同。模型参数前后哈希不变。
- 实际重放入口：`staging/reproduce_official_V28_before_pause/run_replay_revision2.py --actual`；完成证据：`staging/reproduce_official_V28_before_pause/actual_ABC900_mean_replay_revision2/COMPLETE.json`，SHA256 `09a2409ba45023447851215cca2ea6ae42be13e022e28db433f42a40672d684c`。耗时14.11秒，无新训练、θ拟合、计数抽样、评分或官方调用；只读取原 K 公共先验和官方 NTC。
- 初次读取旧 AnnData nullable-string 身份字段失败，已按实际编码修正；第一版代码、绑定和失败记录均保留。科学公式与预设容差未改。
- 原360000细胞完整包已核验一致并保留，本次没有重新生成整个计数包或再次提交。因此这里复现的是原模型前向预测，完整历史计数产物由原包提供。

## 必须区分的两个 V28 基线

**实际正式 V28：** entry `WQ2L6vQomYorxgjkFQNN`，保存的官方 Overall **0.06333542864645086**。使用 V7 tiny Transformer 的统一部署权重，600 步、seed20260928；校准 K562 响应先验＋背景条件残差，再由原 N1 同 donor NB 生成计数。A/B/C 各300靶点、每靶点400细胞，总360000×18533。

**历史 Native V28：** 固定六角色/48目标实例的本地 Overall **0.25932824671778637**，已在此前新 V37 原计数精确重放。该工作流使用历史 V13 reference 的不同外层教师，不能将它当作统一 tiny600 正式部署的本地分数，也不能把它与官方0.0633相减。完整公共背景留出的开发/最终测试仍应另外设计；统一部署已训练 Native 背景，不能冒称整背景留出模型。

## 原方法与可复用文件

- 权重：`archive/legacy_20261003/models/residual_v7/final_candidate/seed_20260928/tiny_transformer_600.pt`，SHA `a2b996eacf498ce63ff584be51d84bc52c77c87ec040f96a8a30d43a91d6dce2`。
- 特征：同目录 `features.npz`，SHA `a6cb72bcdcd0e0a5f01e9bce5a4a09d284fd74924373b69af363eebd694aa372`。
- 原实际导出程序：`archive/legacy_20261003/scripts/submit_v28_20261001_export_v2.py`，SHA `0c0e0f473d909fa48120deb9a6e3b413c86fc59ff9ee1820be5ef069a4f78bc7`。不要直接执行它：它包含旧路径、decoder fit/全包导出，并会调用读取 Native 扰动的 load_data。
- 原部署目录：`archive/legacy_20261003/runs/submission_v28_20261001/`，保留 deployment_lock、prediction_manifest、challenge_effects/context_A/B/C.npz、dispersion.npz、donors_A/B/C.npz 和提交记录。
- 模型轴/core：`archive/legacy_20261003/runs/context_v4/gene_axis.txt`、`normalization_core.txt`；原 cfg 是 `archive/legacy_20261003/configs/baseline_v1.json`。
- K562 原先验银行、STRING 和官方 NTC 原文件仍在当前根 `data/`。原 K 先验与新审计/训练银行的单位和构建不可混用。

原先验为支持量收缩及幅度校准的 raw-count PC.5 log2FC，原网络约99万参数、2层/4头、rank32残差，背景由 NTC 输入。正式 θ 是原 V25 配方 refit 的 gene-specific dispersion；保留相同 donor、原 count_ratio、NB 命名空间、不可编辑坐标的 teacher-conditioned H0。own 不普遍等于原 NTC；六项 cell_eval2 指标均排除自身靶点基因。

## 本会话已完成的结果

| 相同原V28 N1法、固定Native六角色候选 | Overall |
|---|---:|
| 历史V28对照 | 0.25932824671778637 |
| 新U600 | 0.24547042165929026 |
| TxPert raw lookup32＋U600 | 0.2390471304269621 |
| raw均值同靶点MixUp＋U600 | 0.24850263661916028 |
| U600关闭判别项 | 0.2468085084903491 |
| TxPert冻结final graph32＋U600 | 0.24758594486647287 |
| U6000/cos6000 | 0.2524997895221775 |
| U6000只用同context判别负例 | 0.22826993511916513 |

所有均完成实际训练、全六效应、原N1计数和原六指标评分。没有超过固定 V28，不晋升、不按正角色拼装。后面的长训练与同context实验已收尾，无待续训练。

两个认真长训练的新 A/B/C 包已实际提交并published，本服务周期两个新机会已使用。保存的官方R1 **0.057039283529874425**，均匀U **0.05863456459375523**；与旧V28均为 `vcc2026-val-1` / `valA-r4+valB-r4+valC-r4`，两者仍低于V28。本会话不再查询或提交。

## 参考数据和搁置事项

Kinker CPM、Parse全PBS pseudobulk、DepMap24Q2及Tahoe元数据已下载整合；它们用于背景参照，不是原始单细胞计数或CRISPRi真值。来源筛选未得到一致的跨背景收益，勿只凭NTC相关性认定最佳训练源。

TxPert K-only checkpoint、冻结完整图特征和作者单K表达档已保留。最新作者ZIP678,058,077 B、解压H5AD2,839,201,052 B；原5000有序轴已恢复，但 Native 输入只覆盖约80%–84%，完整模型尚未执行。10691作者 K NTC 的固定 processed mean 缓存已实际提取，仅ctrl RNA。full-model U600回退的五个代码草稿已经搁置，不能冒当V28改进或已经完成试验。

原始数据、处理记录和失败保留；最终THP/mod1数值仍未开放。C盘最新剩余约131 GiB，高于50 GiB下限。

详细证据：`reports/V54/HISTORICAL_OFFICIAL_V28_RECIPE_ANCHOR_AUDIT_CN_20261005.md`、同名JSON、`reports/V54/V28_REPRODUCTION_STOP_HANDOFF_CN_20261005.md`、`reports/V54/NATIVE_U_LONG6000_ACTUAL_RESULT_CN_20261005.md`、`reports/V54/NATIVE_WITHIN_CONTEXT_CE_ACTUAL_RESULT_CN_20261005.md`。

用户希望以后在真正 V28 上逐项改进。当前已依最新要求停止；2026-10-05 实际查询持续目标返回 `goal:null`，没有可暂停的活动目标，也未创建新目标。0.35优化目标未达成。当前无本项目 Python 实验进程。新会话只读交接并等待用户；是否继续、先做哪项实验，由用户下一条指示决定。
