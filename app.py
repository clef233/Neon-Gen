import streamlit as st
import pandas as pd
import numpy as np
from generator import DataGenerator
import plotly.express as px
import plotly.graph_objects as go
import io


st.set_page_config(page_title="问卷数据生成器", layout="wide")

def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            st.error('请上传 .xlsx 或 .csv 格式的文件')
            return None
        return df
    except Exception as e:
        st.error(f'文件读取错误: {str(e)}')
        return None

def plot_distribution_comparison(original_data, synthetic_data, column):
    """绘制分布对比图"""
    if pd.api.types.is_numeric_dtype(original_data[column]):
        # 数值型变量使用直方图
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=original_data[column], name='原始数据', opacity=0.7))
        fig.add_trace(go.Histogram(x=synthetic_data[column], name='生成数据', opacity=0.7))
        fig.update_layout(
            title=f'{column} 分布对比',
            barmode='overlay',
            xaxis_title=column,
            yaxis_title='频数'
        )
    else:
        # 分类变量使用条形图
        orig_counts = original_data[column].value_counts(normalize=True)
        syn_counts = synthetic_data[column].value_counts(normalize=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=orig_counts.index, y=orig_counts.values, name='原始数据', opacity=0.7))
        fig.add_trace(go.Bar(x=syn_counts.index, y=syn_counts.values, name='生成数据', opacity=0.7))
        fig.update_layout(
            title=f'{column} 分布对比',
            xaxis_title=column,
            yaxis_title='比例',
            barmode='group'
        )
    
    return fig

def main():
    st.title('问卷数据生成器')
    
    # 初始化生成器
    if 'generator' not in st.session_state:
        st.session_state.generator = DataGenerator()
    
    # 文件上传
    uploaded_file = st.file_uploader("上传数据文件 (XLSX/CSV)", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("数据预览:")
            st.dataframe(df.head())
            
            # 选择分类变量
            st.write("选择分类变量:")
            categorical_columns = st.multiselect(
                "选择分类变量 (非数值型变量)",
                df.columns.tolist(),
                default=[col for col, dtype in df.dtypes.items() if dtype == 'object']
            )
            
            # 训练参数设置
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                epochs = st.number_input("训练轮数", min_value=1, value=500)
            with col2:
                batch_size = st.number_input("批次大小", min_value=1, value=10)
            with col3:
                n_samples = st.number_input("生成样本数", min_value=1, value=len(df))
            with col4:
                pac = st.number_input("PAC值", min_value=1, value=10, help="Privacy Amplification by Conditioning参数，通常设置为1-20之间")
            
            # 训练按钮
            if st.button("训练模型并生成数据"):
                try:
                    with st.spinner('准备数据...'):
                        processed_df = st.session_state.generator.prepare_data(df, categorical_columns)
                    
                    with st.spinner('训练模型中...'):
                        success = st.session_state.generator.train(processed_df, epochs, batch_size)
                        
                    if success:
                        with st.spinner('生成数据中...'):
                            synthetic_df = st.session_state.generator.generate(n_samples)
                            
                        if synthetic_df is not None:
                            st.success('数据生成成功!')
                            
                            # 显示评估结果
                            st.write("### 数据分布对比")
                            
                            # 为每个列创建分布对比图
                            for column in df.columns:
                                fig = plot_distribution_comparison(df, synthetic_df, column)
                                st.plotly_chart(fig)
                            
                            st.write("### 生成数据预览:")
                            st.dataframe(synthetic_df.head())
                            
                        # 下载按钮部分的修改
                        if synthetic_df is not None:
                            # 创建 BytesIO 对象
                            buffer = io.BytesIO()
                            
                            # 将 DataFrame 写入 BytesIO 对象
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                synthetic_df.to_excel(writer, index=False)
                            
                            # 获取字节数据
                            buffer.seek(0)
                            excel_data = buffer.getvalue()
                            
                            # 创建下载按钮
                            st.download_button(
                                label="下载生成的数据 (xlsx)",
                                data=excel_data,
                                file_name="generated_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.error('数据生成失败')
                    else:
                        st.error('模型训练失败')
                        
                except Exception as e:
                    st.error(f'发生错误: {str(e)}')

if __name__ == "__main__":
    main()
