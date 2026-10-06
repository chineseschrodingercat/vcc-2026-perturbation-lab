旧 V28 正式提交确实好于两份新长训练包。更强的离线存档 `archive/legacy_20261003/reports/V28_V115_OFFICIAL_SCORE_DROP_20261003.json` 给出旧 entry `WQ2L6vQomYorxgjkFQNN` 的精确 Overall **0.06333542864645086**；panel `vcc2026-val-1`、anchor `vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4` 与 V53/V54 保存反馈一致。因此 R1 长训包 Δ=-0.006296145116576439，U 长训包 Δ=-0.00470086405269563。最早邮件/截图只有 0.0633，旧目录 `official_status.json` 仍为 launching 是 stale；本次没有调用 API 或查榜。

**本地 0.259328 不是这个官方分数，也不是同一套均值权重。** 本地固定 Native6 V28 使用六个匹配 teacher/V13 effect 资产；正式 ABC V28 则是统一部署的全局 V7 `tiny_transformer_600.pt`，seed20260928。不能拿本地 Native6 数值给旧/新正式包作增量，也不能把 U600/6000 当旧 V28 backbone。

实际 ABC producer 已由 code+manifest 双证实：`archive/legacy_20261003/scripts/submit_v28_20261001_export_v2.py` 的 SHA `0c0e0f473d909fa48120deb9a6e3b413c86fc59ff9ee1820be5ef069a4f78bc7` 与 `prediction_manifest.exporter_sha256` 精确相同。其字面计算为 `base=frozen_v1(targets,genes)`，267 个公共 K n≥30 目标加原 V7 tiny600 residual，33 个无支持目标 residual=0，`effects=base+residual`；对 ABC 只替换原合法 NTC query descriptor。原 prior 只加一次。manifest 同时记录 z/g/h/seen/ownmask 与旧验证 feature parity 全通过。

计数是原 N1：同一 NTC donor raw count×`count_ratio(total_effect,NTC_mean)`，原 6119∩seen 非own NB域及域外 stochastic-round，countseed20260928、donor seed+100；θ来自原 V25 decoder 3000step/seed20261002 的 K/Hep/J/RPE 合法 F234 统一部署 refit。原人口 ABC各300target×400cell、18533gene、总360000cell。

可直接复用的原资产均仍存在，已做 fresh opaque SHA，未解码数组：

| 资产 | 当前归档路径（均在 archive/legacy_20261003/） | 核实情况 |
|---|---|---|
| 原正式包 | runs/submission_v28_20261001/v28_n1_prediction.vcc | 3,145,994,240 B，SHA `f82fb2d796965026fb6ba84e22545c1af0bf78e512a13b7c3bd0f15f2c7c4523`；上传 receipt 同大小、MD5 verified |
| 三份原 ABC total/residual | runs/submission_v28_20261001/challenge_effects/context_{A,B,C}.npz | 三 SHA 逐项匹配原 prediction_manifest；完整 fullmodel轴/300target格式由原 exporter 证实 |
| 原均值 checkpoint、features | models/residual_v7/final_candidate/seed_20260928/ | tiny600 SHA `a2b996…96dce2`、features `a6cb72…daa372`，均与部署锁匹配 |
| 原 θ 表 | runs/submission_v28_20261001/dispersion.npz | gene_symbols/theta 由 producer 证实；当前 SHA `32cf1c31e1467b0d89067fa3bff7ca94efd54bcb374b8c8c5c7febe47b3a88dd` |
| 原 donor rows/UID表 | runs/submission_v28_20261001/donors_{A,B,C}.npz | 三份均存在；完整 SHA 见 JSON，无 donor 数值读取 |

原单独 prediction H5AD 已不在该目录，但完整 `.vcc` 及上述 effect/θ/donor 资产仍在。没有发现预存逐 target `ratio` 数组；未来若需要 ratio，应由原 saved total、原 NTC mean 和冻结 `count_ratio` 字面函数重建，不能补算新的 K prior 冒称 exact复用。本次未重建。

V53/V54 与旧 V28 共用计数法/θ/donor/seed/人口及 267/33 支持回退，但**均值模型不是一项改动**：新模型用 12来源、792target 的新 F23 prepared/共享变换、6000更新、两seed，分别采用 R1 或 family-uniform 日程，并在既定 known6114 非own替换域接回原 total。二者之间的受控变量主要是 q；两者相对旧 V28 同时改变监督来源、预处理资产、训练预算与集成。不能仅凭官方下降判定是“NTC路由失败”或“长训练失败”。旧均值 TRAIN 行的逐项单位/内容本次未重读，只锁其历史训练表 SHA 与原 exporter。

下一主线应保留上述旧 V28 正式均值及原 N1资产作为 anchor，单项改动逐个配对。新候选的域外和低支持坐标从旧 **V28 total** 继承；既有 U600 baseline 不再承担 V28 backbone 的名称。官方最佳包已保留，无需为恢复 baseline 重新提交一次。这里给出可复用事实，不承诺下一项改动必然提升；当前 full Tx 草稿按 Root 新方向停放，未改变模型继续流程。

本次读取八份提交/部署保存记录，另按 Root 明确指针读取一份更强的旧官方 score JSON，并有限读原两份 producer 代码。零网络/API、零 RNA/权重/θ/effect/donor 数值解码、零模型/计数/评分执行。
