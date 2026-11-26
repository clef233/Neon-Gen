"""CPU-only, column-by-column sampling from TabPFN conditional distributions."""
import os

import numpy as np
import pandas as pd

from generator import DataGenerator


class TabPFNGenerator(DataGenerator):
    def __init__(self, version="v3.5-fast", seed=42, context_rows=512, batch_rows=128):
        super().__init__()
        if context_rows < 1 or batch_rows < 1:
            raise ValueError("context_rows 和 batch_rows 必须为正数")
        self.version = version
        self.seed = seed
        self.context_rows = context_rows
        self.batch_rows = batch_rows
        self.context = None

    def prepare_data(self, df, categorical_columns=None):
        # Preserve original category labels, numeric codes and actual missing values.
        self.context = None
        self.categorical_columns = [] if categorical_columns is None else list(categorical_columns)
        self.numerical_columns = [c for c in df if c not in self.categorical_columns]
        for c in self.numerical_columns:
            if not pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                raise ValueError(f"请将 {c} 设为分类变量")
        self.original_data = df
        return df

    def train(self, df):
        if df.empty or not df.columns.is_unique:
            raise ValueError("请提供有记录且列名唯一的数据")
        self.context = df.sample(min(len(df), self.context_rows), random_state=self.seed)
        self.rng = np.random.default_rng(self.seed)
        return True

    def _draw_codes(self, train_x, test_x, codes, categorical_indices):
        """Draw class labels. Partition wide vocabularies into conditional subgroups."""
        labels = np.unique(codes)
        if len(labels) == 1:
            return np.full(len(test_x), labels[0])
        if np.all((train_x == train_x[0]) | (np.isnan(train_x) & np.isnan(train_x[0]))):
            return self.rng.choice(codes, size=len(test_x))

        if len(labels) > 10:
            # Balanced by number of labels. Each leaf only sees its own group's rows.
            groups = np.searchsorted(labels, codes) // int(np.ceil(len(labels) / 10))
            chosen = self._draw_codes(train_x, test_x, groups, categorical_indices)
            result = np.empty(len(test_x), dtype=int)
            for group in np.unique(chosen):
                train_mask, test_mask = groups == group, chosen == group
                result[test_mask] = self._draw_codes(
                    train_x[train_mask], test_x[test_mask], codes[train_mask],
                    categorical_indices,
                )
            return result

        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        import torch

        torch.set_num_threads(1)
        classifier = TabPFNClassifier.create_default_for_version(
            ModelVersion(self.version), device="cpu", n_estimators=1,
            fit_mode="fit_preprocessors", categorical_features_indices=categorical_indices,
            random_state=self.seed,
        )
        classifier.fit(train_x, codes)
        result = np.empty(len(test_x), dtype=int)
        for start in range(0, len(test_x), self.batch_rows):
            block = test_x[start:start + self.batch_rows]
            probs = classifier.predict_proba(block).astype(float)
            cdf = np.cumsum(probs / probs.sum(axis=1, keepdims=True), axis=1)
            cdf[:, -1] = 1.0
            idx = (self.rng.random((len(block), 1)) > cdf).sum(axis=1)
            result[start:start + len(block)] = classifier.classes_[idx]
        return result

    def generate(self, n_samples):
        if self.context is None or n_samples < 1:
            raise ValueError("请先训练模型，并指定正数样本量")
        train_x = np.empty((len(self.context), 0))
        test_x = np.empty((n_samples, 0))
        categorical_indices = []
        generated = {}
        for i, (name, column) in enumerate(self.context.items()):
            categorical = name in self.categorical_columns
            if categorical:
                codes, values = pd.factorize(column, use_na_sentinel=False)
            else:
                numbers = column.to_numpy(dtype=float, na_value=np.nan)
                valid = numbers[~np.isnan(numbers)]
                if len(valid) and not np.isfinite(valid).all():
                    raise ValueError(f"{name} 包含无穷值")
                # Up to nine equal-frequency bins plus an explicit missing category.
                edges = np.unique(np.quantile(valid, np.arange(1, 9) / 9)) if len(valid) else []
                codes = np.where(np.isnan(numbers), -1, np.searchsorted(edges, numbers))
            chosen = self._draw_codes(train_x, test_x, codes, categorical_indices)
            if categorical:
                generated[name] = pd.Series(values.take(chosen), dtype=column.dtype)
                train_feature, test_feature = codes, chosen
                categorical_indices.append(i)
            else:
                donor_rows = np.empty(n_samples, dtype=int)
                for label in np.unique(chosen):
                    mask = chosen == label
                    donor_rows[mask] = self.rng.choice(np.flatnonzero(codes == label), mask.sum())
                generated[name] = column.iloc[donor_rows].reset_index(drop=True)
                train_feature, test_feature = numbers, numbers[donor_rows]
            train_x = np.column_stack((train_x, train_feature))
            test_x = np.column_stack((test_x, test_feature))
        return pd.DataFrame(generated)
