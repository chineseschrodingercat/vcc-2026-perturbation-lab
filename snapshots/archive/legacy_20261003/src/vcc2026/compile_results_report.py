"""Assemble the result report from completed experiment outputs, never guessed scores."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import pandas as pd
import numpy as np

ROOT=Path.cwd()
METRICS=['pds_cosine','expr_mse_unbiased_capped_norm','de_wilcoxon_direction_fidelity_yield_raw',
 'de_wilcoxon_direction_reach_raw','de_wilcoxon_sig_jaccard','de_wilcoxon_lfc_nmae']

def table(df):
    def val(x):
        if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.5f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']+
        ['| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    paths=[Path('runs/model_selection_v2/h1_single_cell_benchmark/six_metric_results.csv'),
       Path('runs/foundation_v3/stack_h1_benchmark/six_metric_results.csv'),Path('runs/foundation_v3/state_h1_benchmark/six_metric_results.csv')]
    for p in paths:
        if not p.exists():raise FileNotFoundError(f'Benchmark unfinished: {p}')
    scores=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    scores=scores[scores.metric.isin(METRICS)]
    wide=scores.pivot(index='model',columns='metric',values='mean')[METRICS].reset_index()
    out=Path('reports');wide.to_csv(out/'all_models_six_metric_comparison.csv',index=False)
    human=wide.rename(columns=dict(zip(METRICS,['PDS ↑','MSE ↓','DE方向可靠度 ↑','DE方向覆盖 ↑','DE Jaccard ↑','LFC误差 ↓'])))
    rawbulk=pd.read_csv('runs/model_selection_v2/h1_confirmation/results.csv')
    bilinear=pd.read_csv('runs/model_selection_v2/bilinear_unseen_targets/test_results.csv')
    training=[]
    for v in ['string','string_source']:
        folder=Path(f'models/foundation_v3/state_{v}')
        cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'complete.json').read_text())
        trace=pd.read_csv(folder/'training_trace.csv');best=trace.loc[trace.validation_loss.idxmin()]
        training.append({'模型':v,'可训练参数':cfg['trainable_parameters'],'训练步数':done['completed_steps'],
          '最佳检查点步数':int(best.step),'验证loss':done['best_validation_loss'],'耗时分钟':done['elapsed_seconds']/60})
    free=shutil.disk_usage(ROOT).free
    # A single filesystem walk records physical file lengths, including caches.
    folders={}
    for top in ['data','models','runs','envs','.cache','reports','src','configs','logs']:
        folders[top]=sum(p.stat().st_size for p in (ROOT/top).rglob('*') if p.is_file())
    storage={'recorded_utc':datetime.now(timezone.utc).isoformat(),'minimum_C_free_GiB':50,'C_free_bytes':free,
       'C_free_GiB':free/2**30,'directory_logical_bytes':folders,'project_recorded_bytes':sum(folders.values()),
       'within_user_limit':free>=50*2**30,'D_used_for_this_project':False}
    if not storage['within_user_limit']:raise RuntimeError('Disk reserve violated')
    (out/'storage_audit_v3.json').write_text(json.dumps(storage,indent=2)+'\n')
    source_registry=json.loads(Path('configs/source_registry_2026-09-27.json').read_text())
    additions=[]
    for p in [Path('data/raw/external/vcc2025_h1/download_manifest.json'),Path('data/raw/external/jiang_mixscale/download_manifest.json'),
              Path('data/raw/external/replogle_2022/k562_gwps/singlecell_download_manifest.json'),
              Path('models/pretrained/stack_large_aligned/checkpoint_manifest.json')]:
        additions.append({'manifest':str(p),**json.loads(p.read_text())})
    source_registry['schema']='vcc2026_local_source_registry_v3';source_registry['additional_files']=additions
    source_registry['source_of_truth_for_transforms']=['data/processed/v1/external_pseudobulk/*.qc.json',
       'data/processed/v2/external_pseudobulk/vcc2025_h1.qc.json','data/processed/v3/state_cells/protocol.json',
       'data/processed/v3/state_cells/input_validation.json']
    source_registry['resource_policy']='configs/resource_policy.json'
    Path('configs/source_registry_v3.json').write_text(json.dumps(source_registry,indent=2)+'\n')
    newer=json.loads(Path('runs/transfer_v2/prediction__normalized_k562_transfer.manifest.json').read_text())
    stackproto=json.loads(Path('runs/foundation_v3/stack_gwps_steps3/protocol.json').read_text())
    lines=[
      '# VCC 2026：第二、三轮模型实测与完整预测交付','',
      f'更新：{datetime.now(timezone.utc).isoformat()}。所有文件位于 `C:\\00 - Competition`。这是实际执行结果；本地公开数据评价与比赛隐藏评分严格区分。','',
      '## 可以直接检查的交付','',
      '- 原基线完整文件：`runs/baseline_v1/prediction__k562_calibrated_direct.h5ad` 和 `.vcc`，参数 `models/baseline_v1/model.json`、`configs/baseline_v1.json`。',
      '- 新的归一化迁移候选：`runs/transfer_v2/prediction__normalized_k562_transfer.h5ad` 和 `.vcc`，参数 `models/transfer_v2/parameters.npz`、`config.json`。归一化空间拟合系数 **0.3728017934**，来源支持收缩 `n/(n+50)`，来源至少 30 个细胞。',
      '- 两份完整预测均为 **360,000 × 18,533**：A/B/C 各 300 靶点，每靶点 400 个细胞；原始整数 UMI counts。新文件有 **2,078,542,900** 个非零值，H5AD **2,917,018,160 bytes**。官方完整格式检查与打包日志位于 `logs/transfer_v2_vcc_prep_*.json`。',
      '- State 两个真实训练模型的权重：`models/foundation_v3/state_string/best.pt` 和 `state_string_source/best.pt`；同目录包含架构、配置、训练曲线和目标级验证划分。',
      '- Stack 原始生成细胞：`runs/foundation_v3/stack_gwps_steps3/generated__<target>.h5ad`；经过计数转换、可与基线配对比较的 H1 预测位于 `runs/foundation_v3/stack_h1_benchmark/`。State 同类文件位于 `state_h1_benchmark/`。',
      '- 本次没有上传比赛服务器，因此没有官方排名或隐藏测试分数。','',
      '**本轮交付选择**：保留原基线作为主候选，归一化版本作为完整备选；暂不把 State/Stack 混合替换为主模型。H1 共同细胞级测试中，局部 raw-scalar 基线的 PDS 为 0.70652，归一化候选为 0.69746；后者的 MSE 为 1.09550，低于基线 1.13065，但仍未低于零效应的 1.03486。State/Stack 及其组合没有同时改善全部六项指标，Stack 直接预测还有明显的背景偏移。这是当前数据与配置下的判断，不是对这些架构能力的普遍否定，也不是隐藏榜单排名预测。','',
      '## 数据增加的理由及体量','',
      '| 数据/模型 | 下载大小 | 实际用途 |','|---|---:|---|',
      '| VCC 2025 H1 训练文件 | 15.48 GB | 新细胞背景的外部验证；221,273 个细胞、150 个扰动、38,176 个 NTC |',
      '| K562 GWPS 原始单细胞 | 65.83 GB | Stack 的真实来源扰动细胞；State 单细胞训练。逐范围续传，完整 MD5 与 SHA-256 核验 |',
      '| Stack-Large-Aligned checkpoint | 2.61 GB | 本机实际 GPU 推理；固定 Hugging Face revision 与 SHA-256 |',
      '| Jiang Mixscale DE 表压缩包 | 0.324 GB | 只检查缺失靶点覆盖；218 个靶点，未补上 28 个缺失靶点，未下载其大型原始数据 |','',
      'H1 与本届面板有 13 个靶点重叠，且只在 H1 中可观察，不是 A/B/C 的真实答案。28 个 K562 缺失靶点经过 Ensembl ID 别名复查及 Mixscale 覆盖检查，仍未获得合格的直接来源；加上 5 个低支持靶点，最终迁移模型明确保留 33 个零效应回退。原始下载均保留，未把未测量基因伪装成生物学零表达。','',
      '## 比较了哪些方法','',
      '1. **12 种跨背景迁移方案**：零效应、原始 log2FC 校准、平均响应、归一化标量校准、对照表达门控、逐基因收缩、中心化逐基因回归、32/64/128 维低秩回归、两种固定混合。',
      '2. **12 组 SLIM 式双线性岭回归**：2 种基线 × 2 种基因基维数 × 3 个正则强度，使用真实 STRING 特征。这是对公开方程的适配，并非论文完整复现。另保留上一轮 ridge/kNN 的未见靶点反例。',
      '3. **State 官方模型类实际训练**：STRING 64 维输入的小模型；STRING 加 64 维 K562 响应特征的较大模型。二者均从头训练，未冒称使用不覆盖本届靶点的现成 one-hot 模型。',
      '4. **Stack 公开大模型实际推理**：3 步 in-context generation；比较直接生成、预测 NTC 校正以及固定 25% Stack + 75% 归一化迁移组合。State 也评估固定 25% + 75% 组合。',
      '5. **计数生成器比较**：原有随机取整，与可令零计数变为非零的 Gamma-Poisson 增量/二项抽稀。后者在本地并未带来全面改善。','',
      '## 验证如何避免泄漏','',
      '迁移模型先按完整目标基因分组，再按整个目的细胞系留出。内层选择只看另外两个背景及内层靶点；外层测试靶点从映射训练中移除。6202 个共同实测基因是开发评价轴。独立 H1 确认使用 6122 个有实测交集的基因，并把全部 150 个 H1 靶点从跨背景映射训练中移除；来源 K562 对这些靶点的观测仍可作为输入，这与任务设定一致。',
      '',
      'State 的 1024 个输入/输出基因仅按开发数据选择；H1 的 150 个靶点标签不参与任何来源的神经网络拟合。其余目标采用确定性哈希留出 10%，同一靶点在所有背景均不进入训练。SVD 基础和标准化也仅在开发目标拟合。H1 扰动细胞完全不进入本地 State 训练。',
      '',
      '需要区分内部验证任务：STRING+K562 分支明确允许输入被验证靶点的 K562 来源响应，因此其内部 loss 不是“从未观察过该扰动”的零样本成绩；其中 K562 组还属于来源背景内预测。它仅用于选择本轮训练检查点，外推判断依赖另行留出的目的 H1。两种 State 输入的信息量不同，内部 loss 下降不能单独证明架构更优。',
      '',
      '24 个细胞级靶点在查看误差前按哈希选定，真实细胞每靶点 400 个，真实 NTC 2000 个与生成用 NTC 池 8000 个互不重叠。所有模型复用相同真实细胞、生成细胞与取整随机种子。只有技术批次与细胞，不将它们声称为独立生物学重复。24 靶点结果仍只是小规模外部试验。',
      '',
      '**Stack 的额外限制**：模型卡报告预训练使用 scBaseCount，alignment 使用 CellxGene/Parse；本次没有逐 accession 证明 H1 与其预训练完全独立。因此只能称本地没有使用 H1 扰动真值，不能将 Stack 结果称为经过预训练去重认证的外部测试。',
      '',
      '## H1 全部 132 个有来源支持靶点：平均表达比较','',
      table(rawbulk[['model','n_targets','relative_mse','pearson','pds_proxy']]),'',
      'relative MSE < 1 才优于零效应。预先在开发数据选中的 hybrid_64 在 H1 上仍为 1.00861，未胜过零效应；虽然其相关性较高，但识别具体靶点的 PDS 下降。这个反例保留，不因组合架构更复杂而忽略。PDS_proxy 是受基因轴和目标集合限制的本地代理。','',
      '## 同一 H1 24 靶点：官方六项指标实现','',
      table(human),'',
      '以上使用 `cell-eval2==0.16.0` 的 `vcc2026` preset、Scanpy Wilcoxon 与真实对照；数值是原始本地指标，未套用榜单 anchor 缩放，不能将六列直接求平均作为比赛总分。LFC NMAE 的官方有效性门槛会使部分低 DE 靶点不计入该项，完整逐靶有效性见各 `metrics__*.csv`。',
      '',
      '表中的 `v1_raw_scalar__round` 为相同方法族在 H1 留出协议下重新拟合的局部基线，基因轴也受 H1 交集约束；并非直接给完整 A/B/C 提交文件打分。归一化候选同理。State/Stack 的均值响应经对照计数转换后才进入该共同评分；Stack 的原始输出另行保留。','',
      '## State 的实际训练记录','',table(pd.DataFrame(training)),'',
      '共同设置：1024 基因、32 细胞/集合、每步 8 个集合、AdamW lr=3e-4、weight_decay=1e-4、cosine 降至 3e-5；损失为官方 energy distance 加 0.1 倍均值 MSE，梯度裁剪 1.0。小模型 hidden=128、2 层、4 heads；较大模型 hidden=256、4 层、8 heads。随机种子 20260927。State 不是经过大规模预训练的完整官方模型复现，本轮容量和训练时长属于可复现的本机初步实验。','',
      f'Stack 本机载入参数量为 {stackproto["parameter_count"]:,}；固定词表 15,012 个基因，来源数据覆盖 {stackproto["source_covered_vocab_genes"]:,} 个。来源未测基因按官方接口填零，但这会带来明显输入缺失限制。batch_size=1、n_cells=512、prompt_ratio=0.25、context_ratio=0.4、float32；Windows 上修复 pandas 字符串兼容，使用原权重。','',
      '## 未见靶点的 SLIM 式结果','',table(bilinear[['group','model','n_targets','relative_mse','pearson','pds_proxy']]),'',
      '开发集选中的 rank=32、lambda=100、训练均值截距模型在 266 个完全未见比赛靶点上 relative MSE=1.00919，未优于零效应；因此不用它强行替换 33 个证据不足靶点的回退。可复用参数已保存到 `models/bilinear_v2/parameters.npz`。','',
      '## 文件管理、复现与使用边界','',
      f'当前记录的项目目录合计约 **{sum(folders.values())/1e9:.2f} GB**，C 盘剩余 **{free/2**30:.2f} GiB**，大于要求的 50 GB；本项目未使用 D 盘。200 GB 总量限制已取消，当前策略见 `configs/resource_policy.json`。逐目录体量见 `storage_audit_v3.json`；数据来源与哈希见 `configs/source_registry_v3.json`。','',
      '运行入口见 `REPRODUCE_V3.md`；两个环境的版本分别固定在 `configs/vcc313.requirements.v2.lock.txt` 和 `foundation312.requirements.lock.txt`。原始文件只读；训练表在 `data/processed/v{1,2,3}/`，参数在 `models/`，逐次预测和指标在 `runs/`。失败日志也保留：Stack pandas 3 字符串错误、bfloat16 到 NumPy 的兼容错误，以及新预测 NPZ 字符串类型错误均已修复并重跑，不把失败运行算作结果。','',
      '后续选择以不同整细胞背景和官方反馈为依据。本轮不宣称 State/Stack 或组合天然胜过简单模型，也不把格式通过称为预测准确。完整基线与归一化候选均保留，便于下一步一次只改变一个模型因素。',
      '',
      '## 核验过的主要来源','',
      '- [VCC 数据页面](https://virtualcellchallenge.org/app/datasets)、[FAQ](https://virtualcellchallenge.org/app/faq)、[Leaderboard](https://virtualcellchallenge.org/app/leaderboard)。',
      '- [cell-eval2 官方六指标说明](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics-brief.md)。',
      '- [State 官方代码](https://github.com/ArcInstitute/state)、[Stack 官方代码](https://github.com/ArcInstitute/stack)、[Stack 模型卡](https://huggingface.co/arcinstitute/Stack-Large-Aligned)。',
      '- [SLIM 官方代码](https://github.com/RasmussenLab/SLIM)、[VCC 2025 官方数据说明](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main/virtual-cell-challenge)。','']
    (out/'MODEL_COMPARISON_V2_V3_CN.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(out/'MODEL_COMPARISON_V2_V3_CN.md'),'models_scored':len(wide),'C_free_GiB':free/2**30},indent=2))

if __name__=='__main__':main()
