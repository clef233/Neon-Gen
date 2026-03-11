import streamlit as st
import pandas as pd
import numpy as np
from generator import DataGenerator
import plotly.graph_objects as go
import io

st.set_page_config(page_title="问卷数据生成器", layout="wide")

# ---- 统一默认参数 ----
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 100
DEFAULT_PAC = 10


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
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=original_data[column], name='原始数据', opacity=0.7))
        fig.add_trace(go.Histogram(x=synthetic_data[column], name='生成数据', opacity=0.7))
        fig.update_layout(
            title=f'{column} 分布对比',
            barmode='overlay',
            xaxis_title=column,
            yaxis_title='频数',
        )
    else:
        # 类别对齐：取两边的并集，缺的补 0
        all_categories = sorted(
            set(original_data[column].dropna().unique()) | set(synthetic_data[column].dropna().unique()),
            key=str,
        )
        orig_counts = original_data[column].value_counts(normalize=True)
        syn_counts = synthetic_data[column].value_counts(normalize=True)

        orig_values = [orig_counts.get(c, 0) for c in all_categories]
        syn_values = [syn_counts.get(c, 0) for c in all_categories]
        cat_labels = [str(c) for c in all_categories]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=cat_labels, y=orig_values, name='原始数据', opacity=0.7))
        fig.add_trace(go.Bar(x=cat_labels, y=syn_values, name='生成数据', opacity=0.7))
        fig.update_layout(
            title=f'{column} 分布对比',
            xaxis_title=column,
            yaxis_title='比例',
            barmode='group',
        )

    return fig


def render_evaluation_table(eval_results):
    """渲染评估指标表格"""
    num_rows = []
    cat_rows = []

    for col, info in eval_results.items():
        if col.startswith('_'):
            continue
        if info['type'] == 'numerical':
            num_rows.append({
                '列名': col,
                '原始均值': round(info['original_mean'], 4),
                '生成均值': round(info['synthetic_mean'], 4),
                '均值差': round(info['mean_diff'], 4),
                '原始标准差': round(info['original_std'], 4),
                '生成标准差': round(info['synthetic_std'], 4),
                '标准差差': round(info['std_diff'], 4),
                '原始范围': f"[{info['original_min']}, {info['original_max']}]",
                '生成范围': f"[{info['synthetic_min']}, {info['synthetic_max']}]",
            })
        else:
            cat_rows.append({
                '列名': col,
                'TVD (总变差距离)': round(info['tvd'], 4),
                '说明': '0=完全一致，1=完全不同',
            })

    if num_rows:
        st.write("**数值列**")
        st.dataframe(pd.DataFrame(num_rows), use_container_width=True)

    if cat_rows:
        st.write("**分类列**")
        st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)

    dup = eval_results.get('_duplicate_rows', 0)
    if dup > 0:
        st.warning(f"生成数据中有 {dup} 行与原始数据完全相同，样本量较小时属正常现象。")
    else:
        st.info("生成数据中没有与原始数据完全相同的行。")


def main():
    st.title('问卷数据生成器')

    if 'generator' not in st.session_state:
        st.session_state.generator = DataGenerator()

    uploaded_file = st.file_uploader("上传数据文件 (XLSX/CSV)", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.write("数据预览:")
            st.dataframe(df.head())

            st.write("选择分类变量:")
            categorical_columns = st.multiselect(
                "选择分类变量 (非数值型变量)",
                df.columns.tolist(),
                default=[col for col, dtype in df.dtypes.items() if dtype == 'object'],
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                epochs = st.number_input("训练轮数", min_value=1, value=DEFAULT_EPOCHS)
            with col2:
                batch_size = st.number_input("批次大小（偶数）", min_value=2, value=DEFAULT_BATCH_SIZE, step=2)
            with col3:
                n_samples = st.number_input("生成样本数", min_value=1, value=len(df))
            with col4:
                pac = st.number_input(
                    "PAC值",
                    min_value=1,
                    value=DEFAULT_PAC,
                    help="batch_size 必须能被 pac 整除",
                )

            # 前端参数校验
            param_ok = True
            if batch_size % 2 != 0:
                st.warning("batch_size 必须为偶数")
                param_ok = False
            if batch_size % pac != 0:
                st.warning(f"batch_size ({batch_size}) 必须能被 pac ({pac}) 整除")
                param_ok = False

            if st.button("训练模型并生成数据", disabled=not param_ok):
                try:
                    with st.spinner('准备数据...'):
                        processed_df = st.session_state.generator.prepare_data(df, categorical_columns)

                    with st.spinner('训练模型中...'):
                        success = st.session_state.generator.train(
                            processed_df, epochs, batch_size, pac
                        )

                    if success:
                        with st.spinner('生成数据中...'):
                            synthetic_df = st.session_state.generator.generate(n_samples)

                        if synthetic_df is not None:
                            st.success('数据生成成功!')

                            # 评估指标表格
                            st.write("### 评估指标")
                            eval_results = st.session_state.generator.evaluate(synthetic_df)
                            render_evaluation_table(eval_results)

                            # 分布对比图
                            st.write("### 数据分布对比")
                            for column in df.columns:
                                fig = plot_distribution_comparison(df, synthetic_df, column)
                                st.plotly_chart(fig, use_container_width=True)

                            st.write("### 生成数据预览:")
                            st.dataframe(synthetic_df.head())

                            # 下载
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                synthetic_df.to_excel(writer, index=False)
                            buffer.seek(0)

                            st.download_button(
                                label="下载生成的数据 (xlsx)",
                                data=buffer.getvalue(),
                                file_name="generated_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                        else:
                            st.error('数据生成失败')
                    else:
                        st.error('模型训练失败')

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f'发生错误: {str(e)}')


if __name__ == "__main__":
    main()
