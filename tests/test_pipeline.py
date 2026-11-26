import unittest

import numpy as np
import pandas as pd

from pipeline import FEATURES, build_lstm_sequences


class LSTMSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = 100
        cls.split = 80
        cls.window = 10
        cls.data = pd.DataFrame(
            {
                feature: np.linspace(1, 2, rows) + index
                for index, feature in enumerate(FEATURES)
            }
        )
        cls.data["target_return"] = np.linspace(-0.01, 0.01, rows)
        cls.data["close"] = np.linspace(100, 200, rows)
        cls.data["date"] = pd.date_range("2020-01-01", periods=rows)

    def test_pca_transforms_sequences_and_fits_train_rows_only(self):
        result = build_lstm_sequences(
            self.data,
            self.window,
            self.split,
            pca_components=6,
        )
        x_train, _, x_test, _, _, _, _, reducer = result

        self.assertEqual(x_train.shape[2], 6)
        self.assertEqual(x_test.shape[2], 6)
        self.assertEqual(reducer.n_samples_, self.split)

    def test_unreduced_sequences_keep_all_features(self):
        result = build_lstm_sequences(self.data, self.window, self.split)

        self.assertEqual(result[0].shape[2], len(FEATURES))
        self.assertIsNone(result[-1])

    def test_pca_rejects_more_components_than_features(self):
        with self.assertRaises(ValueError):
            build_lstm_sequences(
                self.data,
                self.window,
                self.split,
                pca_components=len(FEATURES) + 1,
            )


if __name__ == "__main__":
    unittest.main()
