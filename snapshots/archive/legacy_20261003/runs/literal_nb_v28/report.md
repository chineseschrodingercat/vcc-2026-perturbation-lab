# V28：保留literal V13逐细胞均值，仅改变NB整数采样

唯一主候选 N1 外层本地 overall **0.228136**，对真实匹配 V13 增益 **+0.063137**；独立旧权重重复增益 **+0.059271**。预声明晋级标准 **通过**，三背景整体0.25目标 **尚未超过**。

实际完成 **18 次成功新评分、18 份真实V13评分复用、0 次新拟合**；登记的新评分尝试19次：18次进入冻结judge并成功，1次在生成及judge运行前被磁盘guard拒绝；恢复时另有17次已完成评分的只读核验。inner两臂×六case=12行，outer两臂×十二case=24行。输入预检属于只读来源核验，不增加正式评分预算。0次decoder/encoder/flow forward、0次IPF、0次theta再估计、0次optimizer更新；读取冻结checkpoint中的theta并计算softplus不属于网络forward。这里是已经多轮使用的本机development评估，没有官方提交。

## 唯一冻结比较

H0为原始实际V13的sparse multiplier+stochastic round预测与评分，逐值复用原文件。唯一新主臂N1对同一target、同一实际NTC donor、同一Native gene坐标，定义未round计数 **B_ig=X^NTC_ig·r_tg**，E内保持 **µ_ig=B_ig逐坐标相等**，仅改为NB2(µ,冻结合法role theta)采样。E外复制对应H0的realized integer counts。没有decoder平滑、flow变换、IPF或新均值模型；历史V27 H4没有成为本轮候选。

| arm | 方式 | inner overall | 对真实V13增益 |
| --- | --- | --- | --- |
| H0 | 原始实际V13 sparse round | 0.221516 | +0.000000 |
| N1 | literal V13逐坐标µ＋冻结theta NB | 0.259328 | +0.037812 |

| 目的背景 | H0/V13 | N1 | N1−H0 |
| --- | --- | --- | --- |
| HepG2 | 0.164844 | 0.202613 | +0.037769 |
| Jurkat | 0.400130 | 0.354623 | -0.045507 |
| RPE1 | 0.099574 | 0.220749 | +0.121176 |

inner先每目的背景平均两个合法role，再让三背景等权。规则为N1−H0>0.005、至少2/3背景正向、生成全部有效；实际均增益 +0.037812，正向背景 2/3，generation_valid **通过**，gate **通过**。gate在outer评分前冻结。六case、cells与countseed不是六个新增独立生物学重复。

inner六原始指标、normalized分量、FID分解与计分贡献：

| arm | PDS ↑ | 表达 MSE ↓ | 方向 fidelity/yield (FID) ↑ | 方向 reach ↑ | DE Jaccard ↑ | LFC NMAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.827381 | 1.483025 | 0.428854 | 0.299898 | 0.107006 | 0.836900 |
| N1 | 0.827381 | 1.825045 | 0.587964 | 0.370253 | 0.070889 | 0.849226 |

| arm | norm PDS | norm MSE | norm FID | norm reach | norm Jaccard | norm NMAE | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.727735 | 0.000000 | 0.533856 | -0.100351 | 0.315698 | -0.147842 | 0.221516 |
| N1 | 0.732742 | 0.000000 | 0.832901 | 0.012969 | 0.167790 | -0.190432 | 0.259328 |

| arm | direction precision宏均值 | coverage term宏均值 | 重构FID宏均值 |
| --- | --- | --- | --- |
| H0 | 0.682401 | 0.635451 | 0.428854 |
| N1 | 0.604197 | 0.971832 | 0.587964 |

| 对照 | PDS贡献 | MSE贡献 | FID贡献 | reach贡献 | Jaccard贡献 | NMAE贡献 | 总Δoverall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1−H0 | +0.000835 | +0.000000 | +0.049841 | +0.018887 | -0.024651 | -0.007098 | +0.037812 |

overall是六个normalized component的算术均值，差分量/6相加与总增益已核对。FID为DE direction_fidelity_yield，不是Fréchet距离；逐target precision×coverage之后宏平均，两个宏均值的乘积一般不等于FID。normalized MSE=0表示计分截断分量为零，不表示实际表达误差为零。归一化贡献是评分算术归因，不是生物机制验证。

## 完整背景留出与旧独立权重重复

| arm | 方式 | 主overall | 主V13增益 | 重复overall | 重复V13增益 |
| --- | --- | --- | --- | --- | --- |
| H0 | 原始实际V13 sparse round | 0.165000 | +0.000000 | 0.154056 | +0.000000 |
| N1 | literal V13逐坐标µ＋冻结theta NB | 0.228136 | +0.063137 | 0.213327 | +0.059271 |

| 背景 | arm | 主overall | 主V13增益 | 重复overall | 重复V13增益 |
| --- | --- | --- | --- | --- | --- |
| HepG2 | H0 | 0.134490 | +0.000000 | 0.126509 | +0.000000 |
| HepG2 | N1 | 0.221534 | +0.087044 | 0.199947 | +0.073438 |
| Jurkat | H0 | 0.376648 | +0.000000 | 0.362406 | +0.000000 |
| Jurkat | N1 | 0.356996 | -0.019652 | 0.366555 | +0.004149 |
| RPE1 | H0 | -0.016138 | +0.000000 | -0.026746 | +0.000000 |
| RPE1 | N1 | 0.105880 | +0.122018 | 0.073479 | +0.100224 |

主模型三个countseed先在每背景等权，再让三个背景等权；重复组件权重单独报告。V13 mean-model原训练seed为主20260928/重复20261005，冻结V25 theta组件seed为20261002/20261009。全部来自既有权重，这轮没有重训；countseed是技术抽样。

| 主countseed | N1对V13增益 |
| --- | --- |
| 20260928 | +0.067621 |
| 20260929 | +0.060258 |
| 20260930 | +0.061531 |

N1最终标准：等背景增益 +0.063137>0.005；正背景 2/3≥2；最低主countseed增益 +0.060258>0；旧独立训练权重重复增益 +0.059271>0。

主模型六原始指标、normalized分量、FID和overall贡献：

| arm | PDS ↑ | 表达 MSE ↓ | 方向 fidelity/yield (FID) ↑ | 方向 reach ↑ | DE Jaccard ↑ | LFC NMAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.841270 | 1.334289 | 0.378467 | 0.246203 | 0.059373 | 0.835672 |
| N1 | 0.841204 | 1.670016 | 0.572958 | 0.271721 | 0.052170 | 0.839708 |

| arm | norm PDS | norm MSE | norm FID | norm reach | norm Jaccard | norm NMAE | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.843633 | 0.000000 | 0.257737 | -0.116053 | 0.043118 | -0.038437 | 0.165000 |
| N1 | 0.841223 | 0.000000 | 0.627718 | -0.070202 | 0.022799 | -0.052719 | 0.228136 |

| arm | direction precision宏均值 | coverage term宏均值 | 重构FID宏均值 |
| --- | --- | --- | --- |
| H0 | 0.685872 | 0.588351 | 0.378467 |
| N1 | 0.589155 | 0.973220 | 0.572958 |

| 对照 | PDS贡献 | MSE贡献 | FID贡献 | reach贡献 | Jaccard贡献 | NMAE贡献 | 总Δoverall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1−H0 | -0.000402 | +0.000000 | +0.061663 | +0.007642 | -0.003387 | -0.002380 | +0.063137 |

独立组件权重重复：

| arm | PDS ↑ | 表达 MSE ↓ | 方向 fidelity/yield (FID) ↑ | 方向 reach ↑ | DE Jaccard ↑ | LFC NMAE ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.810119 | 1.290478 | 0.410774 | 0.252815 | 0.042224 | 0.837617 |
| N1 | 0.815873 | 1.625225 | 0.568475 | 0.261234 | 0.050295 | 0.839339 |

| arm | norm PDS | norm MSE | norm FID | norm reach | norm Jaccard | norm NMAE | overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H0 | 0.770009 | 0.000000 | 0.312025 | -0.098886 | -0.011867 | -0.046943 | 0.154056 |
| N1 | 0.783894 | 0.000000 | 0.616937 | -0.084586 | 0.016498 | -0.052782 | 0.213327 |

| arm | direction precision宏均值 | coverage term宏均值 | 重构FID宏均值 |
| --- | --- | --- | --- |
| H0 | 0.709773 | 0.603496 | 0.410774 |
| N1 | 0.589238 | 0.965555 | 0.568475 |

| 对照 | PDS贡献 | MSE贡献 | FID贡献 | reach贡献 | Jaccard贡献 | NMAE贡献 | 总Δoverall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N1−H0 | +0.002314 | +0.000000 | +0.050819 | +0.002383 | +0.004727 | -0.000973 | +0.059271 |


## 逐坐标均值、零坐标与实际采样

完整generation_targets历史顺序、countseed+100 donor RNG、原pool count mean、pseudocount=0.5、倍率上限16与fallback规则均沿用V13。pseudocount只属于已有ratio计算；没有向B添加正pseudo-rate。H0仍按旧sparse非零存储顺序和[countseed,target顺序index,0]的round stream验证。N1 NB stream采用预声明namespace，相同seed不构成不同µ下NumPy rejection sampler的逐基因底层随机数完全配对主张。

E为input6119∩role-seen∩own-gene mask∩Native measured genes；own、unseen和外轴逐整数保护。theta来自合法V25 role的既有decoder最终checkpoint gene-specific softplus(raw theta)+theta_floor，只读取该参数；inner只用合法剩余Native监督，outer目的Native扰动整个背景已在该旧fit中排除。V13 effect继承历史NPZ的真实panel target序与10916 model gene轴，没有伪造内嵌IDs，并验证effect=V1+saved residual。真实目的扰动truth只进入冻结judge，不用于选择µ、theta、mask或draw。

**raw donor count=0时，B=0且N1严格为0**，本轮没有恢复raw dropout或激活全pool零诱导基因。B>0但H0 round恰好为0时，N1可能抽到正值，那是已有正期望坐标的sampling差异。NB不会学习target-specific dropout或新的基因相关结构。

对于固定donor，Var(Y|B)=B+B²/theta、P0=(theta/(theta+B))^theta；跨donor variance为Var_D(B)+E_D[B+B²/theta]。条件独立NB的off-diagonal raw covariance期望仍来自Cov_D(B_g,B_h)，增加的对角噪声会改变correlation、zero fraction与actual library。literal donor counts已含观测噪声，而theta原本配合smoothed decoder µ拟合；转用B可能重复加计数噪声而过度离散。不能把采样收益外推为普遍生物学结构改善。

E内raw expected逐坐标/row/column均值固定，**实际有限NB样本的gene mean、library和非线性scorer normalization后的mean不必相等**。完整条件期望使用E内B+E外已经固定的H0整数，不能拿E外未round B代替。实际draw必须是非负有限精确float32整数，最大count≤2^24、row library<10^6；guard失败保存原draw并停止，不clip、不重抽。

读取 18 个 N1 新生成job、232 条 target×job 保留诊断；H0没有重新生成。逐目标µ/B数组与SHA完全一致、保护区域计数SHA一致，µ=0理论variance=0/P0=1，raw零坐标没有新增非零。每job目标先等权，再每背景等权；主countseed等权，main/repeat分别列示。这些是机制核验条目，不能当作新增独立生物学样本数。

| 阶段 | target×job条目 | E最少基因 | E最多基因 | literal零µ坐标总数 | raw零新增非零 | theta最小 | theta最大 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inner | 48 | 6118 | 6119 | 28191445 | 0 | 9.522321 | 10.482539 |
| outer_main | 138 | 6118 | 6119 | 80141308 | 0 | 9.525794 | 10.486753 |
| outer_repeat | 46 | 6118 | 6119 | 26723469 | 0 | 9.520963 | 10.475420 |

| 阶段 | E实际mean MAE对B | E理论P0 | E理论gene var含donor异质性 | native实际P0 | H0实际P0 | native实际gene var | H0实际gene var |
| --- | --- | --- | --- | --- | --- | --- | --- |
| inner | 0.071687 | 0.651682 | 27.315964 | 0.662430 | 0.595470 | 29.463818 | 24.672211 |
| outer_main | 0.073436 | 0.649766 | 26.668871 | 0.660976 | 0.593569 | 28.575796 | 23.735594 |
| outer_repeat | 0.072943 | 0.649606 | 27.109514 | 0.660699 | 0.593232 | 28.942564 | 24.180676 |

| 阶段 | actual library MAE对H0 | actual library平均差 | library相关H0 | 完整条件mean MAE |
| --- | --- | --- | --- | --- |
| inner | 142.567741 | -1.504795 | 0.999705 | 0.048131 |
| outer_main | 145.765492 | -1.520152 | 0.999684 | 0.049252 |
| outer_repeat | 145.585621 | +2.391653 | 0.999688 | 0.048921 |

理论P0/variance只列E内；actual native矩包括未编辑坐标，两者范围不同。完整条件mean MAE使用E内B与E外fixed realized H0构成的正确期望，不把未round protected B当成当前条件期望。JSON/NPZ中的G0_*是同donor实际V13 H0的兼容字段别名，未混入V25 flowmean baseline。0µ坐标总数是多job机制记录之和，不能解释为独立样本规模。

机械诊断只比较生成计数与其条件期望/同donor H0，没有另做目的真实扰动covariance error或scorer-normalized marginal fidelity诊断。性能证据来自上面保留的六指标judge表，不由P0/variance机械改善代替。

## 独立验证及制品

V28 completion audit **通过**。报告只读取实际保留six raw metrics、normalized components、FID、原始H0和新生成diagnostics，重新聚合gate、paired差与预算；当前重新核对 725 份冻结source哈希。旧V27报告和failed H3 gate保持原结论；历史设计依据为 [next_test_review.md](<C:/00 - Competition/runs/v13_nb_v27/next_test_review.md>)，不是把V27探索性H4换名重做。

N1符合本轮预声明development晋级条件。没有新training、theta postfit、copula、architecture fit、云计算、外部数据下载或官方提交。已反复使用的本地开发背景不能当作新外部验证。

可复核制品：[decision.json](<C:/00 - Competition/runs/literal_nb_v28/decision.json>)、[inner_gate.json](<C:/00 - Competition/runs/literal_nb_v28/inner_gate.json>)、[inner_six_metrics.csv](<C:/00 - Competition/runs/literal_nb_v28/inner_six_metrics.csv>)、[outer_six_metrics.csv](<C:/00 - Competition/runs/literal_nb_v28/outer_six_metrics.csv>)、[inner_jobs.json](<C:/00 - Competition/runs/literal_nb_v28/inner_jobs.json>)、[outer_jobs.json](<C:/00 - Competition/runs/literal_nb_v28/outer_jobs.json>)、[outer_started.json](<C:/00 - Competition/runs/literal_nb_v28/outer_started.json>)、[protocol_lock.json](<C:/00 - Competition/runs/literal_nb_v28/protocol_lock.json>)、[score_calls.json](<C:/00 - Competition/runs/literal_nb_v28/score_calls.json>)、[generator_preflight.json](<C:/00 - Competition/runs/literal_nb_v28/generator_preflight.json>)、[audit.json](<C:/00 - Competition/runs/literal_nb_v28/independent_audit/audit.json>)。


## 资源恢复、调用计数与时间记录

第19条登记的新评分尝试是末个原case的同命令重试。此前第18条登记尝试在75GiB空闲磁盘guard处停止，尚未产生该case的新count block或进入judge；前17份已完成的新评分、全部原始H0、模型权重和冻结输入保持原值。恢复时对这17份已有evaluation启动只读scorer核验，不重复生成或judge。实际scorer子进程共36次=19次新尝试+17次已有结果只读核验，pipeline.log中的35条ready与1次guard失败及冻结runner执行路径已对应核对；正式新judge仍只有18次。

恢复记录 [disk_recovery.json](<C:/00 - Competition/runs/literal_nb_v28/disk_recovery.json>)（SHA256 `378be71fa652b2e18413ba0f75cfcfa8a1caef8897926375d49b6aeb71e173cf`）记载失败调用与原日志、17组已完成本轮job中独立cache_real副本的逐文件hash/bytes清理。仅移除68份可从原来源恢复的缓存副本，合计68.79MiB；其原文件和保留结果再次核对通过。运行时另有真实可用空间波动，见 [disk_recovery_runtime_observations.json](<C:/00 - Competition/runs/literal_nb_v28/disk_recovery_runtime_observations.json>)（SHA256 `514b68a8c063e092c2cc9c2e571cc59220e2cb69246733c80505a5449bba1097`）；其原因未确定，观察到系统pagefile增长不能当成因果解释。因此不把全部可用空间变化归因缓存清理，重试仍使用原75GiB guard。没有删除保留结果、改变预声明gate、重抽既有counts或新增拟合。

原评分首次开始至最终decision完成的**记录UTC跨度为20.11分钟**，不能称为精确实际wall time。原journal中HepG2独立旧权重case的结束UTC比开始早16.227936秒，原始记录和首次严格时间审计失败均保留：[audit_failure.json](<C:/00 - Competition/runs/literal_nb_v28/independent_audit/audit_failure.json>)（SHA256 `d4624e4ada5c6d59ef8522af5a5b1d31f00a49b4c660baacd18666739f28346f`）。只读Windows事件证据 [clock_event_evidence.json](<C:/00 - Competition/runs/literal_nb_v28/clock_event_evidence.json>)（SHA256 `a16275a05466441d03d9dbdf406e46e6b19ff0d7c1c51927b76c79f76ce53fe9`）确认一次Kernel-General ID1系统时间回拨61.056257秒；根本原因未确定，我们没有改系统时间或设置。若假设没有其他未记录的调整，按该次回拨校正的约21.13分钟只是一项推算，不能无条件当成精确整轮耗时。decision中的202.61秒仅为恢复runner的局部monotonic elapsed。

原时间记录没有修写。新的独立审计恢复只对这条精确非单调duration使用冻结subprocess.run顺序、原journal、pipeline/gate执行路径替代该时间断言，其余科学、计数、来源和指标检查重新运行且保留。恢复manifest [audit_clock_recovery_manifest.json](<C:/00 - Competition/runs/literal_nb_v28/independent_audit/audit_clock_recovery_manifest.json>)（SHA256 `23631e1d6d4c01211bc22cf4ca8bcc3dc2b1590401162310732373cbb856a415`）与NEW审计wrapper [complete_audit_literal_nb_v28_clock.py](<C:/00 - Competition/scripts/complete_audit_literal_nb_v28_clock.py>)（SHA256 `8373ac06fdf5e414d81bc94ff9e192a2099076e907e1b564c4a312871377b885`）可复核这一精确例外。整个本轮仍有18次成功新judge和18份H0复用；登记的19次新评分尝试中，额外1次是judge前资源拒绝，时钟审计恢复没有产生新训练、生成或评分。

冻结报告源保持不变；本独立finalizer仅修正调用层级与耗时说明，所有评分、均值/保护证据及判定仍由冻结报告和completion audit复核。
