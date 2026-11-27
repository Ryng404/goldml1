import unittest

import numpy as np
import pandas as pd

from pipeline import (
    FEATURES,
    build_lstm_sequences,
    clean_data,
    infer_gap_threshold,
    train_linear,
)


class LSTMSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = 100
        cls.split = 90
        cls.validation_split = 80
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
        cls.data["target_date"] = cls.data["date"].shift(-1)

    def test_pca_transforms_sequences_and_fits_train_rows_only(self):
        result = build_lstm_sequences(
            self.data,
            self.window,
            self.split,
            pca_components=6,
            validation_split=self.validation_split,
        )
        x_train, _, x_val, _, x_test, _, _, dates, _, _, reducer = result

        self.assertEqual(x_train.shape[2], 6)
        self.assertEqual(x_val.shape[2], 6)
        self.assertEqual(x_test.shape[2], 6)
        self.assertEqual(reducer.n_samples_, self.validation_split)
        self.assertEqual(pd.Timestamp(dates[0]), self.data.loc[self.split, "target_date"])

    def test_unreduced_sequences_keep_all_features(self):
        result = build_lstm_sequences(
            self.data,
            self.window,
            self.split,
            validation_split=self.validation_split,
        )

        self.assertEqual(result[0].shape[2], len(FEATURES))
        self.assertIsNone(result[-1])

    def test_pca_rejects_more_components_than_features(self):
        with self.assertRaises(ValueError):
            build_lstm_sequences(
                self.data,
                self.window,
                self.split,
                pca_components=len(FEATURES) + 1,
                validation_split=self.validation_split,
            )

    def test_scaler_is_fitted_only_on_train_rows(self):
        data = self.data.copy()
        data.loc[self.validation_split:, FEATURES] = 1_000_000
        result = build_lstm_sequences(
            data,
            self.window,
            self.split,
            validation_split=self.validation_split,
        )
        scaler = result[-2]

        np.testing.assert_allclose(
            scaler.mean_,
            self.data.iloc[:self.validation_split][FEATURES].mean().to_numpy(),
        )

    def test_linear_forecasts_and_naive_baseline_use_next_price_date(self):
        _, scores, output = train_linear(
            self.data,
            self.split,
            validation_split=self.validation_split,
        )
        test_index = self.split

        self.assertEqual(len(output["dates"]), len(self.data))
        self.assertEqual(
            pd.Timestamp(output["dates"][test_index]),
            self.data.loc[test_index, "target_date"],
        )
        self.assertEqual(
            output["naive_predicted"][test_index],
            self.data.loc[test_index, "close"],
        )
        self.assertIn("RMSE", scores)
        self.assertIn("MAPE", output["baseline_metrics"])


class CleaningTests(unittest.TestCase):
    def test_timestamps_are_sorted_deduplicated_and_flat_candles_kept(self):
        raw = pd.DataFrame(
            {
                "date": [
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                ],
                "open": [103, 100, 101, 102],
                "high": [103, 100, 101, 102],
                "low": [103, 100, 101, 102],
                "close": [103, 100, 101, 102],
                "volume": [1, 1, 1, 1],
            }
        )

        cleaned, report = clean_data(raw)

        self.assertTrue(cleaned["date"].is_monotonic_increasing)
        self.assertEqual(cleaned["date"].duplicated().sum(), 0)
        self.assertIn(100, cleaned["close"].to_numpy())
        self.assertIn(102, cleaned["close"].to_numpy())
        self.assertIn("1 duplicate timestamps", report.iloc[2]["details"])

    def test_gap_threshold_marks_only_large_time_discontinuities(self):
        dates = pd.Series(pd.to_datetime([
            "2020-01-01",
            "2020-01-01 00:15",
            "2020-01-01 00:30",
            "2020-01-10",
        ], format="mixed"))

        self.assertEqual(infer_gap_threshold(dates), pd.Timedelta(days=3))


if __name__ == "__main__":
    unittest.main()
