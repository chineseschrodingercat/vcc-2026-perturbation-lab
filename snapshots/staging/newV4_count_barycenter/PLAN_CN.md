# V4 revision2：计数空间 barycenter，预登记三个权重

2026-10-04。原 V4 code/artifacts 已通过 newV4_carrier_readout/V4_FROZEN_MANIFEST_COMPLETE.json 冻结，此独立adapter不修改原renderer。初次manifest写入只覆盖了一个文件，已保留该不完整记录，以COMPLETE文件为准。

固定三个 arms：w=[0.25,0.50,0.75]，所有背景共用相同规格，不根据当前每背景score选权重、不删任何panel target。

两个端点用同一个原Native job、旧V28 effects、donor rows和generation order：

- N1：原V28 literal NB，theta_scale=1，不用已搜索的theta multiplier。
- posterior carrier：已冻结V4 posterior_augmentation renderer。

编辑域仍原6119∩seen∩nonown；两个端点protected均为同一次namespace0的原H0，所以该块在mix中直接继承H0，不进行新随机化。

计数空间混合 M=wY_N1+(1−w)Y_carrier；在编辑域执行 Y=floor(M)+Bernoulli(M−floor(M))，独立固定 namespace [count_seed,target_generation_index,45,0]。w=0或1为机械端点分支，直接返回原端点，不消费namespace45 RNG。

这是每个cell/gene count坐标的barycenter，与旧V122按200/200 donor metadata选择完整NB/H0 cells不同。所有最终值整数非负并满足原float32≤2^24、row library∈(0,1e6)界限；失败记录，不裁剪或重抽。

条件于两个realized端点，随机舍入的E[Y]=M；条件于donor与冻结状态，E[Y]=wXrho+(1−w)E[Y_carrier]。完整pool随机uniform donor的人口期望仍μrho。posterior endpoint的选定subset均值本就不等于literal Xrho，混合不恢复此等式。null时w>0仍含原N1背景再采样，只有w0严格X；不声称所有权重null恒等。

不声称方差、协方差、gene/row library或realized pseudobulk精确锁定。原N1 namespace26、V4 namespace44、round namespace45分开；同donor共享observed X，两个端点不独立于donor，不能把它们的population covariance写成0。不同w共用相同round namespace，实现相同位置的uniform阈值配对，但不同w可有不同floor/fracs，不宣称integer count逐坐标同号改变。

机械pilot：三背景真实首target×全部预登记权重，w0/w1与两个原renderer逐字节相等；保护块/整数/row guard；完整pool解析conditional means验证人口μrho。固定小256×256真实count块、8MC seeds检查floor+Bernoulli条件均值与解析SE。不加载held perturbation truth，不运行judge。
