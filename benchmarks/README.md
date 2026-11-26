# GSS 问卷合成基准

用真实问卷检验两件事：各题选项的分布能保留多少，以及同一受访者不同答案之间的关系能保留多少。所有方法用同一组训练行和独立留出行。

## 数据

`gss_cat.csv.gz` 是 NORC 美国 General Social Survey 的教学摘录，覆盖 2000–2014 年的 8 个调查年份、21,483 条记录和 9 列。CSV 从 forcats 的 R 数据文件直接导出，保留原始标签、行序和缺失。压缩文件 SHA-256 为 `77ba4529d4cc95937b4fdcc6ac7de6e6b7c6f37fa855d7de0655880d672dba49`。

[变量说明](https://forcats.tidyverse.org/reference/gss_cat.html) · [原始 R 文件（固定提交）](https://github.com/tidyverse/forcats/blob/f83e0e682d9f874d066630ff78eb12586b5b2a32/data/gss_cat.rda) · [摘录整理代码](https://github.com/tidyverse/forcats/blob/f83e0e682d9f874d066630ff78eb12586b5b2a32/data-raw/gss-cat.R) · [forcats 版权声明](forcats-LICENSE.md)

## 实验设计

- 原项目的 CTGAN 需要完整数值输入，故共同使用年龄和看电视时长都有回答的 11,299 行。保留收入“拒答”、宗教“None”等原始选项。结果描述该子样本，未加调查权重。
- 随机种子 17、29、43。每次无放回抽 512 行训练、1,024 行留出，生成 512 行。种子间独立抽取，每个种子内所有方法共用同一划分。按照 CSV 原列序合成。
- 数值变量为年龄和每天看电视时长，其余 7 列按类别处理。项目默认 CTGAN 训练 500 轮，批量 100，PAC 10。TabPFN 每步使用一个估计器，数据和模型均在本机，每次条件预测最多 128 行。数值目标最多分为 9 个分位箱，并单列缺失；在预测箱内抽取原始数值。高于 10 类的分类目标逐层预测类别组和组内类别。
- CART 对照逐列拟合决策树，叶节点至少 12 个样本，在预测叶内随机抽取真实答案。逐列独立抽样保留边际分布，整行重抽样保留真实受访者的完整答案。
- 边际 TV 是各列总变差距离的均值，双变量 TV 是 36 个变量对的联合分布总变差距离均值。评估阶段用训练集五分位点划分数值列。两种距离均为 0 时代表分布相同。另列宗教与教派的联合 TV，以检查问卷中紧密关联的两道题。
- 下游任务是用其他 8 个变量预测受访者是否已婚。逻辑回归分别在合成数据与等量真实训练数据上拟合，在相同真实留出集上计算 ROC AUC。重合率指合成行中有多少能在训练集找到完全相同的原始记录。
- Linux、Python 3.12、AMD EPYC 9V74、CPU 单线程。每个方法和种子都由新进程运行。计时包括后端导入、拟合与采样，排除数据读取、评分和首次权重下载。峰值为从进程启动到采样结束的 Linux 最大 RSS，包含 Python 与已加载的库。未测 Streamlit 多用户并发。

## 复现

在仓库根目录运行。首次使用新版需先在 Prior Labs 接受许可并配置 `TABPFN_TOKEN`，或将已获授权的权重按原文件名放在 `TABPFN_MODEL_CACHE_DIR` 中。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python benchmarks/survey.py --output benchmark-results.json
```

只复跑新版可使用 `--methods tabpfn-v3.5-fast tabpfn-v3.5`。只复跑公开对照可使用 `--methods independent bootstrap cart ctgan tabpfn-v2`。任一运行失败时脚本保留逐次结果并以非零状态退出。

## 2026-11-26 实测

质量和耗时取三个随机种子的均值。内存取三次中的最高值。未经合成的真实训练集在同一留出集的平均 AUC 为 0.6401。

| 方法 | 边际 TV ↓ | 双变量 TV ↓ | 宗教 × 教派 TV ↓ | 是否已婚 AUC ↑ | 原行重合率 | 平均耗时（秒） | 峰值 RSS（MiB） |
|---|---:|---:|---:|---:|---:|---:|---:|
| TabPFN 3.5 Fast | 权重待授权 | — | — | — | — | — | — |
| TabPFN 3.5 | 权重待授权 | — | — | — | — | — | — |
| TabPFN v2 | 0.0651 | 0.1556 | 0.1156 | 0.5912 | 0.00% | 33.3078 | 639 |
| 项目 CTGAN | 0.1412 | 0.2621 | 0.4733 | 0.5582 | 0.00% | 65.4989 | 421 |
| 逐列 CART | 0.0650 | 0.1561 | 0.1224 | 0.6065 | 0.13% | 0.7592 | 157 |
| 逐列独立抽样 | 0.0624 | 0.1824 | 0.5312 | 0.4885 | 0.00% | 0.0009 | 95 |
| 整行重抽样 | 0.0670 | 0.1639 | 0.1396 | 0.6213 | 100.00% | 0.0004 | 95 |

同一套逐列合成方法用 v2 公开权重跑通了三个划分。v2 比默认 CTGAN 更能保留变量间的关系；与逐列 CART 的双变量 TV 接近，预测婚姻的平均 AUC 略低。整行重抽样会直接复制受访者，所以单独保留其重合率作为对照。

3.5 标准版和 Fast 各尝试三个划分，均在首次加载时收到官方 SDK 的 TabPFNLicenseError。等待已授权权重后复跑，才能判断 3.5 的质量、时延和内存。全新 Python 3.12 环境通过依赖检查；网页上传、选后端、生成与导出按钮已覆盖。3.5 的成功界面测试使用无需条件推理的常量前缀，真实权重缺失时另行验证页面会报错。Streamlit Community Cloud 尚未实机部署。

[逐次数据和错误记录](results.json) · [TabPFN 官方实现](https://github.com/PriorLabs/TabPFN) · [TabPFN-4 预告及候补状态](https://priorlabs.ai/technical-reports/tabpfn-4-preview)
