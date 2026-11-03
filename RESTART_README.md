# 2026-11-03 快速重启

项目位置：`E:\VCC2026_Restart_20261103`。工作分支：`restart-20261103`。

本轮已实际恢复数据、模型资产、环境和一条可运行的跨背景预测路线。历史归档保留原样；新代码集中在 `restart/`。目前尚未获得官网登录后的 validation/final 控制数据，因而还没有生成官方 final `.vcc`，也没有新增任何比赛提交。

## 现在怎么用

1. 双击根目录 **`LOGIN_AND_GET_OFFICIAL_DATA.cmd`**。在已登录 Chrome 中打开 [Account → Credentials](https://virtualcellchallenge.org/app/credentials)，将 VCC API key 粘贴到本地 CLI 的隐藏提示处。不要把 key 发到聊天或写入代码。脚本登录后先下载 final，再下载 validation；中断后可重新运行。
2. 官方输入准备好后，双击 **`BUILD_FINAL.cmd`**。默认使用 measured-union 候选，在真实官方基因轴上重新校准、生成 D/E/F 整数计数，并调用本地官方 `vcc prep`。首次真实官方 bundle 仍须通过这一步的输入和格式检查。
3. 双击 **`PROJECT_STATUS.cmd`** 查看数据、模型和生成文件的实际状态。

构建输出放在新的 `outputs/restart/final_<UTC时间>_union/`，包含 `final_prediction.vcc`、`build_receipt.json` 和每一步日志。失败不会伪造成功状态或覆盖旧 run。默认入口不上传比赛文件，因此不会消耗提交次数。

可选独立候选：

```powershell
.\.venv\Scripts\python.exe restart\build_final.py --recipe core
.\.venv\Scripts\python.exe restart\build_final.py --recipe v28 --device auto
```

`v28` 是重建的历史推理回退路线，不能视为新模型的独立留出验证。需要重新下载外部数据时运行 `RESUME_DOWNLOADS.cmd`；已有文件按下载记录校验后复用。

## 已完成的真实工作

| 项目 | 实际状态 |
|---|---|
| 原 GitHub 仓库 | 克隆自 `chineseschrodingercat/vcc-2026-perturbation-lab`，起点 `a7a3491c71c99bd0c393caf445ed689858f6e240` |
| Python / CUDA | 本项目 `.venv` 基于已有 `bigbro` 科学环境共享包；Python 3.13.9、Torch 2.9.0+cu128，CUDA 可用 |
| 官方 CLI | `vcc-cli 0.3.0`，已检查实际命令与格式契约 |
| 下载 | 12/12 文件完成，共 21,651,990,519 bytes，约 21.65 GB |
| 完整性 | 所有下载记录实际大小和 SHA256；11 个文件另核对已有发布方/原始记录哈希，CD4T 文件只有本地 SHA256 与预期大小校验 |
| 旧资产 | 恢复 V28 checkpoint / features / dispersion，以及 Clean400 release 资产 |
| 数据处理 | 656,343 个真实单细胞及三份 construct-mean 数据，形成 6 个 source banks |
| 生物学背景 | K562、RPE1、HepG2、Jurkat 共 4 个；同背景的不同文件不算独立背景 |
| 新模型开发 | 两个版本均完成 4 背景留出，每背景 512 targets，共 2,048 个 target-context 评估 |
| 计数生成 | 已用 11,485 个真实 RPE1 NTC 测试：2 targets × 400 cells，800 × 8,749 CSR 整数矩阵，所有检查通过 |
| 官方输入 | validation 与 final 均尚未登录下载；真实面板覆盖率及 final 文件尚待检查 |

原项目保存的 V28 官方 Overall 为 **0.06333542864645086**，它与历史 Native/Clean400 本地评估口径不同。不能把本地约 0.26–0.27 的数字当作官方成绩。旧 archive 的 7 个 milestone、1,521 个文件引用已通过完整性检查。

## 已下载数据及用途

| 数据 | 用途与边界 |
|---|---|
| Replogle K562 GWPS、K562 essential、RPE1 raw_bulk | 已用于预测。矩阵为每细胞 construct 均值，按 `num_cells_filtered` 加权，不当作单细胞整数计数 |
| scPerturb HepG2、Jurkat、RPE1 | 已用于预测；实际 `.X` 为 dense raw counts，按整数计数校验并聚合 |
| SLIM STRING gene embeddings | 恢复旧 V28 推理所需特征 |
| Feng iPSC genome-wide 与 targeted 汇总签名 | 已下载和审计；genome-wide 共 43,187,656 target-gene 行、6,673 targets、6,472 genes。其协变量调整后的 effect 不能直接当作原始 log2FC 混入模型，暂作下一步候选特征 |
| Primary CD4 T genome-scale Perturb-seq DE 统计 | 已下载 16.79 GB；33,983 条 condition-target 记录、11,526 unique targets、10,282 genes、三个刺激状态。已完成 metadata/schema 审计，尚未将整块 effect 矩阵读入训练 |
| Legacy V28 / Clean400 模型资产 | 已恢复；大模型文件保留本地，代码和小型报告由 Git 管理 |

下载的准确 URL、大小和哈希要求在 `restart/download_manifest.json`，候选来源与选择理由在 `restart/external_candidates.json`。额外数据解释见 `restart/EXTRA_DATA.md`。CD4T 的 `.X` 是合法 null dataset，效果存放在 layers，不能按 raw count H5AD 读取。

## 新的轻量路线与实际结果

先从真实 source 构造扰动 pseudobulk，再用目标背景的 NTC 均值衡量背景相似度，进行收缩后的跨背景迁移。每个生物学背景只投一票；重叠文件按支持细胞数选取。对于没被数据测量的基因，保留目标背景的控制均值。

保留两个候选：

- **Core**：仅预测 6 份来源共同测量的 6,544 个基因。
- **Measured union**：共同基因用于归一化，预测范围扩展到来源并集的 11,920 个基因；每个基因仅聚合实际测过它的来源。该数量尚未与官方 18,533 基因轴取交集。

以下是 **pseudobulk proxy MSE，排除扰动靶基因自身**；每个背景等权：

| 评估基因面板 | 零效应基线 | 选中的迁移候选 | 同面板 MSE 降幅 |
|---|---:|---:|---:|
| Core | 0.02382104 | 0.02128246 | 10.66% |
| Measured union | 0.02492029 | 0.02307768 | 7.39% |

两个版本均选中 alpha=0.75、temperature=4。它们的评估基因集合不同，不能通过这两行的绝对值判定哪一个更好。超参数也使用这些留出折选择，因此这些数字是开发调参结果，**不是独立泛化估计，更不是官方六指标评分**。Frozen V28 历史上见过这些背景，未混入这项“未见背景”比较。

选择默认 union 的理由是保留可测量基因覆盖；真实 final 控制文件到位后，需要检查其目标覆盖、基因轴和背景相似度。更大的数据量或更复杂的模型不自动保证更高比赛分数。

## 资源与时间安排

本机为 128 GB RAM、Ryzen 7 9800X3D（8 核 / 16 线程）和 RTX 5080。用户已允许充分使用 CPU、GPU 与 RAM，并授权本项目最多占用 1 TB E 盘空间。

资源入口采用约 **100 GiB RAM 工作预算**，同时保留至少 16 GiB 可用内存；按实际空闲内存下调。单任务最多 12 个数值计算线程，重新聚合数据可同时处理 4 个来源；final count generation 默认 6 个有界并行采样任务。HDF5 由主线程顺序写入，避免并发损坏；不会同时持有 900 个目标的全部稠密计数。V28 默认自动选择 CUDA，统计迁移和 I/O 使用 CPU。内存按有用数据与中间矩阵需求分配，不为提高占用率而创建无用缓存。

本轮实际项目约 22.12 GB。1 TB 是使用上限；目前优先完成面板匹配、可靠导出和候选检查。已有处理无需因资源设置提高而重跑。

已在本机用真实 RPE1 控制均值和背景留出后的预测均值，测量 12 targets × 400 cells 的 Poisson 计数生成。按 1、6、6、1 workers 顺序各测两次：串行中位耗时 **1.349 秒**，6 workers 为 **0.369 秒**，该环节约 **3.66 倍**；所有 CSR 输出摘要完全一致。并行测试采样到的进程 RSS 峰值约 **2.20 GB**。这里包括计数校验与摘要计算，不包括 HDF5 压缩写盘，也不代表默认 NB 路径或完整 final 的加速倍数。记录见 `reports/count_worker_benchmark.json`。

## 官方时间与提交约束

根据 [Arc 2026 公告](https://arcinstitute.org/news/virtual-cell-challenge-2026) 和 [官方 CLI Guide](https://vcc-cli-wiki.virtualcellchallenge.org/)：

- 截止时间：**2026-11-05 23:59 UTC**，即北京时间 **2026-11-06 07:59**。
- Validation 为 A/B/C；final 为另外三个细胞背景 D/E/F，且 perturbation panel 不同。
- 官方提供各背景 NTC 和目标列表，扰动后的 ground truth 不公开。
- Final 输出含 3 backgrounds × 300 targets × 400 cells = 360,000 cells，18,533 genes，必须为非负、有限、整数 raw counts；不能提交 NTC。
- Final 盲评，每 UTC 日最多两次，最后一次 final 提交用于计分。自动构建只完成本地 prep。

## 文件索引与验证边界

- `restart/lean_transfer.py` / `lean_union.py`：数据聚合、背景留出、预测和计数输出。
- `restart/build_final.py`：真实 final 输入到本地 `.vcc` 的入口。
- `restart/legacy_v28_adapter.py`：重建历史 V28 均值推理，保存原始资产/轴校验。
- `restart/runtime_profile.py`：当前硬件资源分配与线程配置。
- `restart/project_status.py` 与 `reports/RESTART_STATUS.json`：实际文件与实验状态。
- `reports/real_count_smoke.json`：真实 RPE1 计数导出检查。
- `reports/recovered_v28_smoke.json` / `recovered_v28_gpu_smoke.json`：CPU/GPU 资产加载和真实输入推理检查；不是新的生物学验证。
- `restart/benchmark_count_workers.py` / `reports/count_worker_benchmark.json`：真实输入上的并行采样性能与完全一致性核验。
- `outputs/restart/core_cv/` 与 `union_cv/`：实际逐目标开发结果、模型参数、输入文件锁。
- `logs/`：任务日志。历史失败尝试可能仍在追加日志内；最终状态以 receipt / report 为准。

只有输入结构和代码契约已经检查；没有对尚未下载的官方 final 文件声称完成端到端验证。拿到登录后 bundle，运行构建才会解析其真实数据和验证 `.vcc`。
