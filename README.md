 

# Neon-Gen 问卷数据生成器

这是一个基于 CTGAN（Conditional Tabular GAN，条件对抗生成网络）的问卷数据生成工具
提供*在线演示*
https://neon-gen-ajmybve5pc69dvcnt8arvg.streamlit.app/

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
- Python 3.9
- 依赖库见 `requirements.txt`

**免责声明：**

本工具旨在用于生成模拟的问卷数据，以支持研究和开发。**严禁**将本工具用于任何形式的学术造假或不道德行为。使用者对其行为负全部责任。


