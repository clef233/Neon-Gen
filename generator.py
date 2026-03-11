import pandas as pd
import numpy as np
from ctgan import CTGAN
import warnings
warnings.filterwarnings('ignore')


class DataGenerator:
    def __init__(self):
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.integer_columns = []
        self.original_data = None
        self.column_ranges = {}

    def prepare_data(self, df, categorical_columns=None):
        """准备训练数据"""
        if categorical_columns is None:
            categorical_columns = []

        self.original_data = df.copy()
        self.categorical_columns = list(categorical_columns)
        self.numerical_columns = [col for col in df.columns if col not in self.categorical_columns]

        df_processed = df.copy()

        # 分类列：填充缺失值，统一转字符串
        for col in self.categorical_columns:
            df_processed[col] = df_processed[col].fillna("缺失").astype(str)

        # 数值列：检测哪些是整数列，记录范围
        self.integer_columns = []
        self.column_ranges = {}
        for col in self.numerical_columns:
            non_null = df_processed[col].dropna()
            if len(non_null) > 0 and non_null.apply(lambda x: float(x).is_integer()).all():
                self.integer_columns.append(col)
            self.column_ranges[col] = {
                'min': df_processed[col].min(),
                'max': df_processed[col].max(),
            }

        return df_processed

    def train(self, df, epochs=500, batch_size=100, pac=10):
        """训练模型"""
        # 参数校验
        if batch_size % 2 != 0:
            raise ValueError("batch_size 必须为偶数")
        if batch_size % pac != 0:
            raise ValueError(f"batch_size ({batch_size}) 必须能被 pac ({pac}) 整除")

        try:
            self.model = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                pac=pac,
                verbose=True,
            )
            # CTGAN 直接接受原始分类列，不需要手动编码
            self.model.fit(df, discrete_columns=self.categorical_columns)
            return True
        except Exception as e:
            print(f"训练错误: {str(e)}")
            return False

    def generate(self, n_samples):
        """生成新数据"""
        try:
            if self.model is None:
                raise Exception("请先训练模型")

            synthetic_data = self.model.sample(n_samples)

            # 数值列后处理
            for col in self.numerical_columns:
                col_min = self.column_ranges[col]['min']
                col_max = self.column_ranges[col]['max']
                # 先 clip 到原始范围
                synthetic_data[col] = synthetic_data[col].clip(col_min, col_max)
                # 整数列转整数，其余保留两位小数
                if col in self.integer_columns:
                    synthetic_data[col] = synthetic_data[col].round(0).astype(int)
                else:
                    synthetic_data[col] = synthetic_data[col].round(2)

            return synthetic_data

        except Exception as e:
            print(f"生成错误: {str(e)}")
            return None

    def evaluate(self, synthetic_data):
        """评估生成数据与原始数据的差异"""
        results = {}

        for col in self.original_data.columns:
            if col in self.numerical_columns:
                orig = self.original_data[col].dropna()
                syn = synthetic_data[col].dropna()
                results[col] = {
                    'type': 'numerical',
                    'original_mean': orig.mean(),
                    'synthetic_mean': syn.mean(),
                    'mean_diff': abs(orig.mean() - syn.mean()),
                    'original_std': orig.std(),
                    'synthetic_std': syn.std(),
                    'std_diff': abs(orig.std() - syn.std()),
                    'original_min': orig.min(),
                    'synthetic_min': syn.min(),
                    'original_max': orig.max(),
                    'synthetic_max': syn.max(),
                }
            else:
                orig_dist = self.original_data[col].value_counts(normalize=True)
                syn_dist = synthetic_data[col].value_counts(normalize=True)
                all_cats = set(orig_dist.index) | set(syn_dist.index)
                tvd = 0.5 * sum(
                    abs(orig_dist.get(c, 0) - syn_dist.get(c, 0)) for c in all_cats
                )
                results[col] = {
                    'type': 'categorical',
                    'original_distribution': orig_dist.to_dict(),
                    'synthetic_distribution': syn_dist.to_dict(),
                    'tvd': tvd,
                }

        # 重复行检查
        merged = pd.merge(self.original_data, synthetic_data, how='inner')
        results['_duplicate_rows'] = len(merged)

        return results
