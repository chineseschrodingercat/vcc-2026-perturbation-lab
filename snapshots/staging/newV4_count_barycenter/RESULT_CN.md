# V4 revision2 计数空间 barycenter：机械验证完成

三个预登记权重 w_N1=[0.25,0.50,0.75] 对所有背景相同。端点是原V28 N1(theta_scale=1)与已冻结V4 posterior carrier，使用完全相同job/donor/旧effects。每个cell/gene的计数作凸组合，独立namespace45 floor+Bernoulli整数化；不根据本轮每背景结果选择权重。

三背景真实首target的全部九个mixed blocks已经生成并检查，未写完整评分panel、未运行judge。

- 机械w0/w1分支精确返回两个原renderer CSR data/indices/indptr字节。
- 保护域逐坐标保留原H0；全部整数、非负、float32 exact-integer和row-library界限通过。
- 使用完整3000 NTC pool的解析conditional means证明人口期望mu*rho；本轮最大误差4.26e-14。
- 逐坐标floor+Bernoulli的8次技术MC检查另存在mechanical_pilot/complete.json。不同权重共用namespace45的uniform位置配对；原N1/V4/舍入分别namespace26/44/45。没有假定两个population端点独立。
- 原V4共51个source/artifact文件(43.62MB)冻结哈希全部保持；不完整第一次manifest保留，完整清单以V4_FROZEN_MANIFEST_COMPLETE.json为准。

均值是w×literal Xrho+(1−w)×carrier conditional mean；选定donor subset仍不精确等于literal Xrho。方差、协方差、gene/row library和realized pseudobulk未锁定；w>0的null仍含N1再采样，只有w0恒等X。

该方法与旧V122 whole-cell 200/200 splice不同：同一个cell/gene count坐标内混合两个完整端点，再unbiased整数化。实际Overall需root按冻结Native replicate合同评分，不能从机械验证推断已经突破。

调用入口native_adapter.py：generate_case(job,out,name,weight)，实际weight仅接受0.25/0.5/0.75；BarycenterPanelGenerator额外接受w0/w1仅作机械端点检查。原V4renderer独立载入，避免两个native_adapter模块的缓存混用。CPU2/GPU0，不读held perturbation truth或官方score。
