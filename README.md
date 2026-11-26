# 近期正在重构和清理代码，希望引入更多的方法并保持生成的鲁棒性和有效性

# Neon-Gen 问卷数据生成器

本项目用于合成问卷数据。此实验分支保留默认后端 CTGAN，并提供 CPU 本地推理的 TabPFN 3.5 Fast 和 3.5 标准版。
提供*在线演示*
https://neon-gen-ajmybve5pc69dvcnt8arvg.streamlit.app/

## 原论文：
Modeling Tabular data using Conditional GAN 	
arXiv:1907.00503 [cs.LG] 
https://doi.org/10.48550/arXiv.1907.00503
NeurIPS 2019

## 功能特点

- 📊 支持多种数据类型的处理（数值型、分类型）
- 🎯 保持原始数据的统计特性（大概）
- 🖥️ 友好的 Web 界面，基于 Streamlit 开发
- 📈 实时数据预览和下载功能
- 🤤 基于条件对抗生成网络🤤
- P   C   R 扩增（确信）

## 模型参数说明

### 批次大小（Batch Size）
- **必须为偶数**
- 默认值：100

### pac=10
- pac=10
### 训练轮次（Epochs）
- 推荐范围：100 - 1000
- 默认值：500

### 生成数量
- 可根据实际需求设置

## 使用方法

1. 准备数据
   - 支持 CSV ,xlsx格式
   - 确保数据已经过基础清洗
   - 建议移除敏感信息

2. 上传数据
    - 点点按钮
    
3. 设置参数
   - 选择非数值列
   - 调整模型参数
   - 设置生成数量

4. 开始生成
   - 点击"开始生成"按钮
   - 等待模型训练完成
   - 下载生成的数据

## 部署要求

### 环境要求
- 在 Streamlit Community Cloud 的 Advanced settings 选择 Python 3.12，入口为 `app.py`。Linux x86-64 使用 `requirements.txt` 中固定的 CPU 版 PyTorch。
- 在 [Prior Labs](https://ux.priorlabs.ai) 接受模型许可后，可从 [官方 Hugging Face 仓库](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/main) 下载所选版本的权重，保留原文件名，放在 Linux 默认目录 `~/.cache/tabpfn/` 或 `TABPFN_MODEL_CACHE_DIR` 指定的目录。已缓存权重可在本地 CPU 推理。若使用 SDK 自动下载，可将 `TABPFN_TOKEN` 配置为环境变量或 Streamlit 顶层 secret。
- 后端按文件列顺序生成。首列从经验分布抽取，后续列根据已生成的答案预测并抽样。数值列按分位数分箱，再从选中箱内抽取观察值。分类列保留选项，缺失值可被抽中。
- CPU 默认取最多 512 行作为模型上下文，每批预测 128 行。基准结果、数据来源和复现命令见 [GSS 问卷合成实验](benchmarks/README.md)。
  


## **免责声明：**

本工具旨在用于生成模拟的问卷数据，以支持研究和开发。**严禁**将本工具用于任何形式的学术造假或不道德行为。使用者对其行为负全部责任。


