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

本次实测在接受许可后，从 [Prior Labs 的 Hugging Face 仓库](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/main) 下载了下列权重。SHA-256 与仓库记录一致。保留文件名，放入 Linux 默认目录 `~/.cache/tabpfn/` 或 `TABPFN_MODEL_CACHE_DIR` 指定的目录后，在项目根目录执行测试。

| 权重文件 | 字节数 | SHA-256 |
|---|---:|---|
| `tabpfn-v3.5-fast-20260909.safetensors` | 334181708 | `14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550` |
| `tabpfn-v3.5-20260909.safetensors` | 876027932 | `ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3` |

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
| TabPFN 3.5 Fast | 0.0656 | 0.1521 | 0.1400 | 0.5826 | 0.13% | 44.5548 | 2264 |
| TabPFN 3.5 | 0.0637 | 0.1550 | 0.1309 | 0.5818 | 0.07% | 110.9827 | 4235 |
| TabPFN v2 | 0.0651 | 0.1556 | 0.1156 | 0.5912 | 0.00% | 33.3078 | 639 |
| 项目 CTGAN | 0.1412 | 0.2621 | 0.4733 | 0.5582 | 0.00% | 65.4989 | 421 |
| 逐列 CART | 0.0650 | 0.1561 | 0.1224 | 0.6065 | 0.13% | 0.7592 | 157 |
| 逐列独立抽样 | 0.0624 | 0.1824 | 0.5312 | 0.4885 | 0.00% | 0.0009 | 95 |
| 整行重抽样 | 0.0670 | 0.1639 | 0.1396 | 0.6213 | 100.00% | 0.0004 | 95 |

在这份 GSS 样本上，3.5 Fast 的平均双变量 TV 为 0.1521，标准版为 0.1550，CART 为 0.1561，三者均优于 CTGAN 的 0.2621。在合成问卷上训练、真实问卷上预测婚姻状况时，CART 的 AUC 为 0.6065，高于两个 3.5 版本。整行重抽样的 AUC 更高，但生成记录全部复制自训练集。三个随机划分仅报告描述性比较。

免费 CPU 部署优先考虑逐列 CART。Fast 生成 512 行平均约 45 秒，峰值 2.21 GiB；标准版平均约 111 秒，峰值 4.14 GiB。Streamlit Community Cloud 文档列出的内存上限为 2.7 GB。标准版在此设置下已超过该值，Fast 的数字仅包括基准进程，实际应用还需要容纳页面和用户会话。[Streamlit 资源说明](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

两个 3.5 版本各完成三个随机划分。另在本地用 Streamlit AppTest 上传含变化数值列和分类列的 CSV，通过两个版本的真实权重完成生成，并验证生成后有下载按钮。未在 Streamlit Community Cloud 实机部署。

[逐次实验结果](results.json) · [TabPFN 官方实现](https://github.com/PriorLabs/TabPFN) · [TabPFN-4 预告及候补状态](https://priorlabs.ai/technical-reports/tabpfn-4-preview)
