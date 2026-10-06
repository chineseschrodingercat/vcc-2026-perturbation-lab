# KRR 固定三种子配对验证：结果与下一轮决定

2026-10-05 14:55 UTC。本地实验完成且独立核验，仍属反复调参的 Native 开发集；未打开最终 THP 数值、未查询官方分数、未提交。

主要读数：KRR overall 0.271666379388，对照 clean B0 原 NB 0.267185041654，配对增益 +0.004481337734。它是当前最高的已核实、未晋升的三种子 clean 候选。三个种子的总体增益均为正，但背景×种子的描述性 95% 区间 [-0.018461699966, +0.027277959769] 跨零；JURKAT 三种子均退步。因此不能认定稳定跨背景改进，也未达到 clean 0.30/0.35 里程碑。

| count seed | clean B0 NB | KRR NB | 配对变化 |
|---|---:|---:|---:|
| 20260928 | 0.270590981844 | 0.275481072081 | +0.004890090237 |
| 20260929 | 0.271148123067 | 0.274017819711 | +0.002869696644 |
| 20260930 | 0.259816020051 | 0.265500246371 | +0.005684226320 |

| 完整留出背景 | clean B0 NB 三种子均值 | KRR NB 三种子均值 | 配对变化 |
|---|---:|---:|---:|
| HepG2 | 0.228403887646 | 0.233200600975 | +0.004796713329 |
| JURKAT | 0.366085785342 | 0.347330065454 | -0.018755719888 |
| RPE1 | 0.207065451974 | 0.234468471735 | +0.027403019761 |

| 六项 replicate-scaled | clean B0 NB | KRR NB |
|---|---:|---:|
| PDS | 0.657395214070 | 0.670303141425 |
| MSE | 0 | 0 |
| Direction Fidelity | 0.802825369896 | 0.809460245635 |
| Direction Reach | 0.059366405000 | 0.082498065126 |
| DE Jaccard | 0.172388828787 | 0.175535858276 |
| LFC NMAE | -0.088865567829 | -0.107799034136 |

与首种子不同，三种子平均 PDS 也改善；NMAE 退步完整保留。唯一主要晋升指标仍为 overall，不要求六项同时上涨。所有原 48 验证对象、各指标固有 real-side gate 和缺值均保留。种子只衡量整数计数随机性，不是生物重复；同背景两 role 也不是两个独立研究。

协议和产物绑定：
- 评估 fingerprint c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c；原 400 / 全轴 / 固定真实扰动及 score NTC / 原 anchors / 六项各1/6。
- PROTOCOL 22901dd0a9998cf161836d2303f0263f17f82877f8c0cd1604de4cb8e1edfa7a。
- PREDICTIONS_SEALED 8bed547560cdc296971ead26aa14ddd5af60cb5aab34454715f0e3015b704f0c，在所有新 judge 之前封存。
- GENERATION_INDEPENDENT_AUDIT 85bb62247077bcbbeb5aa1b408d14c35e5e3f050bc13c3bba6ad35cde6ee6f42，96新＋96同seed B0/H0逐整数重放 exact。
- COMPLETE 68d45b1e940fedaca6d978ededc85571ac4deca3febabe303f9126a4d323a235，root scorer session60102 EXIT0，全36logical／12新judge／24严格reuse。
- POSTSCORE_INDEPENDENT_AUDIT 4086eafbfd9864f88fbccc69468908fc02d27ae5cb0a909b76524e9130044ea6，36×6 raw/两scaled/MSE ratio-of-sums/原anchors/宏均与独立 exact729 crossed CI全PASS；1124 opaque SHA起止核验，最大raw误差2.67e-15。无新judge/count/fit/RNA。

下一轮继续研究，不停止：不扫 kernel λ、bandwidth、来源或幅度来挽救本轮。优先补齐 KRR 专属、完整来源组件留出的方向概率验证契约；另一次 TRAIN 小预算 shared-vs-specific 联合 EB pilot 将与旧逐源收缩、raw均值和zero同人口比较。后者是交换性新组件响应假设，不冒称 NTC 已识别新载荷，不将旧 SE 接到 canonical public19。

一次 KRR+B0 等权 postdecoder rawconditional-profile 混合可作为独立探索草稿。此前 V62/V76 混合负结果必须保留；本轮资格来自新的 public19 KRR 三种子总体重复正增益及与B0的响应差异，并非混合机制新颖。仅固定1/2，不扫权重、训练或生成器；是否有总体收益须从实际整数计数完整评分判断。原单候选/混合的 full-axis CP 均值、LFC、显著性和 MSE 均不假定保持。

全部原始、正式V28、clean B0、历史失败及本轮产物保留。C盘约123.72GiB。研究 goal 持续 active。