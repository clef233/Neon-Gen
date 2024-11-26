import pandas as pd
import numpy as np
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataGenerator:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.original_data = None
        
    def prepare_data(self, df, categorical_columns=None):
        """准备训练数据"""
        if categorical_columns is None:
            categorical_columns = []
        
        # 保存原始数据用于评估
        self.original_data = df.copy()
        
        # 存储列信息
        self.categorical_columns = categorical_columns
        self.numerical_columns = [col for col in df.columns if col not in categorical_columns]
        
        # 对分类变量进行编码
        df_processed = df.copy()
        for column in categorical_columns:
            le = LabelEncoder()
            df_processed[column] = le.fit_transform(df_processed[column].astype(str))
            self.label_encoders[column] = le
            
        return df_processed
    
    def train(self, df, epochs=500, batch_size=10,pac=2):
        """训练模型"""
        try:
            self.model = CTGAN(
                epochs=epochs,
                batch_size=batch_size,
                pac=pac,
                verbose=True
            )
            
            self.model.fit(
                df,
                discrete_columns=self.categorical_columns
            )
            return True
        except Exception as e:
            print(f"训练错误: {str(e)}")
            return False
    
    def generate(self, n_samples):
        """生成新数据"""
        try:
            if self.model is None:
                raise Exception("请先训练模型")
                
            # 生成样本
            synthetic_data = self.model.sample(n_samples)
            
            # 将分类变量转换回原始标签
            for column in self.categorical_columns:
                le = self.label_encoders[column]
                synthetic_data[column] = le.inverse_transform(synthetic_data[column].astype(int))
                
            # 对数值列进行四舍五入
            for column in self.numerical_columns:
                synthetic_data[column] = synthetic_data[column].round(2)
                
            return synthetic_data
            
        except Exception as e:
            print(f"生成错误: {str(e)}")
            return None
            
    def get_column_stats(self, df, column):
        """获取列的基本统计信息"""
        stats = {}
        if column in self.numerical_columns:
            stats['mean'] = df[column].mean()
            stats['std'] = df[column].std()
            stats['min'] = df[column].min()
            stats['max'] = df[column].max()
        else:
            value_counts = df[column].value_counts(normalize=True)
            stats['distribution'] = value_counts.to_dict()
        return stats

    def evaluate(self, synthetic_data):
        """简单评估生成的数据"""
        evaluation = {}
        
        # 对每一列进行评估
        for column in self.original_data.columns:
            original_stats = self.get_column_stats(self.original_data, column)
            synthetic_stats = self.get_column_stats(synthetic_data, column)
            
            if column in self.numerical_columns:
                evaluation[column] = {
                    'original': original_stats,
                    'synthetic': synthetic_stats
                }
            else:
                evaluation[column] = {
                    'original_distribution': original_stats['distribution'],
                    'synthetic_distribution': synthetic_stats['distribution']
                }
                
        return evaluation
