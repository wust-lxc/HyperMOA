# 聚类评价口径

## 正式文件

- 工作簿：项目根目录 `benchmark _Clustering.xlsx`。
- 基础指标：`data/evaluation/fair_benchmark/clustering_slice_metrics_fair.csv`。
- 补充空间指标：`data/evaluation/fair_benchmark/additional_clustering/`。
- 论文五指标表：`data/evaluation/fair_benchmark/clustering_paper_metrics_slice.csv` 和 `clustering_paper_metrics_macro.csv`。
- 论文图片与绘图代码：`figure/clustering/`。

论文当前展示 ARI、AMI、Moran's I、PAS、CHAOS。综合分是全部五项的
等权归一化排名均值，PAS、CHAOS 越低越好，其余越高越好。
并列使用平均排名；排名使用完整精度，展示时保留三位小数。
NMI 和其他辅助指标仍保存在基础或补充评价结果中，不纳入当前五项综合分。

## 统一评价集合

同一切片的七种算法使用共同的有效 spot 集合：真实标签有效，且七种
预测聚类标签均非缺失。匹配使用原始 barcode，不根据结果好坏筛选 spot。
Human 真实标签统一大小写及 medulla vessel(s) 拼写；缺失标签不计入评价。
ARI、NMI、AMI 不依赖预测簇编号，也不使用绘图时的颜色匹配。

Moran's I 使用原始空间坐标的六近邻，对每个簇的 one-hot 指示变量计算，
再按簇的 spot 数加权；PAS、CHAOS 的具体定义见补充评价脚本与其 manifest。
空间平滑并不等同于正确的解剖分区，需与外部标签指标共同解释。

数据集总体结果是切片指标的等权平均，而不是将独立切片的簇编号强行
合并后评价。确实具有跨切片一致簇编号的方法另有 pooled 补充表，不能
将其混用为每个方法的总体结果。

## 聚类配置与绘图

Human 请求 10 类，Simulation 请求 5 类，Mouse Brain 请求 18 类。
请求类别数与实际非空簇数需区分。Human 的正式 HyperMOA 聚类输入为
`HyperMOA_bc`，mclust 请求 G=10，实际全局有 9 个非空簇，保留原配置。
各算法独立运行设置由对应 runner、上游仓库和已有实验记录提供，评价
代码不会重新聚类或统一替换它们的模型。

当前展示切片为 Human slice2、Simulation slice5、Mouse Brain H3K27ac。
这是已选定切片的展示，不代表独立随机抽样。图片只做标签配色和显示
坐标变换，不写回聚类标签，也不改变计算指标的坐标。

## 代码

- 基础评价：`scripts/evaluation/clustering/evaluate_fair_clustering.py`。
- 补充评价：`scripts/evaluation/clustering/evaluate_additional_clustering_metrics.py`。
- 五项综合分与发布：`scripts/publication/sync_clustering_paper_metrics.py`。

以上路径相对于 `benchmark/`。复现命令见 [reproduction.md](reproduction.md)。
