import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tabpfn_generator import TabPFNGenerator


class GeneratorTests(unittest.TestCase):
    def test_constant_prefix_and_missing_values(self):
        data = pd.DataFrame({"blank": [np.nan] * 6, "code": pd.array([1, 2, None] * 2, dtype="Int64")})
        generator = TabPFNGenerator()
        generator.train(generator.prepare_data(data, ["code"]))
        result = generator.generate(200)
        self.assertTrue(result.blank.isna().all())
        self.assertEqual(str(result.code.dtype), "Int64")
        self.assertEqual(set(result.code.dropna()), {1, 2})
        self.assertTrue(result.code.isna().any())
        self.assertEqual(generator.evaluate(result)["_duplicate_rows"], 200)

    def test_batching_large_vocab_and_determinism(self):
        from tabpfn import TabPFNClassifier

        class EmpiricalClassifier:
            def fit(self, x, y):
                self.classes_, counts = np.unique(y, return_counts=True)
                self.probs = counts / counts.sum()
                assert len(counts) <= 10

            def predict_proba(self, x):
                assert 1 <= len(x) <= 16
                return np.tile(self.probs, (len(x), 1))

        data = pd.DataFrame({"age": np.arange(200), "group": ["g" + str(i % 23) for i in range(200)],
                             "value": np.tile([np.nan, 0.5, 1.25, 10.75], 50)})
        generator = TabPFNGenerator(context_rows=80, batch_rows=16)
        with patch.object(TabPFNClassifier, "create_default_for_version",
                          side_effect=lambda *args, **kwargs: EmpiricalClassifier()) as create:
            generator.train(generator.prepare_data(data, ["group"]))
            result = generator.generate(100)
            generator.train(generator.prepare_data(data, ["group"]))
            pd.testing.assert_frame_equal(result, generator.generate(100))
            self.assertEqual(create.call_args.kwargs["device"], "cpu")
            self.assertEqual(create.call_args.args[0].value, "v3.5-fast")
        self.assertEqual(len(generator.context), 80)
        self.assertEqual(list(result), list(data))
        self.assertTrue(result.age.isin(generator.context.age).all())
        self.assertTrue(result.value.isin(data.value).all())
        self.assertGreater(result.group.nunique(), 10)
        self.assertTrue(result.group.isin(data.group).all())
