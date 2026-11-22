"""验证特征、样本支撑和回测的防泄漏约束。"""

from datetime import date
import unittest

from exchange_tool.backtest import walk_forward
from exchange_tool.data import read_bundle
from exchange_tool.labels import chain_groups, label_all
from exchange_tool.model import (
    BaseRateModel,
    BackoffModel,
    extract_features,
    fit_model,
    summarise_predictions,
)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")

    def test_features_ignore_reviewed_reach(self):
        loan = self.bundle.loans["L019"]
        features = extract_features(self.bundle, loan)
        # 借出时 reach=104，复核补录 corrected_reach=109；特征只能用借出值，
        # 104 距湖站混合用途器具上限 103 仅差 1 厘米，109 则差 6 厘米。
        self.assertEqual(features.size, "near_range")
        self.assertEqual(loan.reach, 104)
        self.assertNotIn("corrected_reach", features.as_dict())

    def test_unknown_doorway_is_its_own_bucket(self):
        features = extract_features(self.bundle, self.bundle.loans["L132"])
        self.assertEqual(features.width, "unmeasured")


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")
        cls.asof = date(2026, 12, 22)
        cls.labels = label_all(cls.bundle, cls.asof)

    def test_unsupported_config_is_sent_to_manual_measurement(self):
        model = fit_model(self.bundle, self.labels, self.asof, model="backoff", min_support=8)
        loan = self.bundle.loans["L140"]
        prediction = model.predict("L140", extract_features(self.bundle, loan))
        self.assertFalse(prediction.supported)
        self.assertIsNone(prediction.risk)
        self.assertEqual(prediction.status, "manual_measurement_required")

    def test_supported_config_has_interval(self):
        model = fit_model(self.bundle, self.labels, self.asof, model="backoff", min_support=8)
        loan = self.bundle.loans["L147"]
        prediction = model.predict("L147", extract_features(self.bundle, loan))
        self.assertTrue(prediction.supported)
        self.assertIsNotNone(prediction.risk)
        self.assertIsNotNone(prediction.interval)
        self.assertLessEqual(prediction.interval[0], prediction.risk)
        self.assertGreaterEqual(prediction.interval[1], prediction.risk)

    def test_training_excludes_non_trainable_labels(self):
        model = fit_model(self.bundle, self.labels, self.asof, model="backoff", min_support=8)
        trainable = sum(1 for label in self.labels.values() if label.trainable)
        self.assertEqual(model.training_size, trainable)

    def test_higher_min_support_reduces_coverage(self):
        low = fit_model(self.bundle, self.labels, self.asof, model="backoff", min_support=5)
        high = fit_model(self.bundle, self.labels, self.asof, model="backoff", min_support=20)
        covered_low = sum(
            1
            for request in self.bundle.requests
            if low.predict(request, extract_features(self.bundle, self.bundle.loan(self.bundle.request(request).loan))).supported
        )
        covered_high = sum(
            1
            for request in self.bundle.requests
            if high.predict(request, extract_features(self.bundle, self.bundle.loan(self.bundle.request(request).loan))).supported
        )
        self.assertGreater(covered_low, covered_high)


class BacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")

    def test_walk_forward_does_not_reuse_chain_across_sides(self):
        chains = chain_groups(self.bundle)
        result = walk_forward(
            self.bundle,
            cutoffs=[date(2026, 9, 1)],
            model="backoff",
            min_support=8,
            horizon_days=60,
        )
        fold = result["folds"][0]
        self.assertGreater(fold["test_loans"], 0)
        # 测试集中不应出现训练链：用个案编号取反查借用链。
        labels = label_all(self.bundle, date(2026, 9, 1))
        training_chains = {
            chains[loan_id]
            for loan_id, label in labels.items()
            if label.trainable and label.observed_on is not None and label.observed_on <= date(2026, 9, 1)
        }
        for case in result["cases"]:
            self.assertNotIn(chains[case["loan"]], training_chains)

    def test_excluded_statuses_are_reported(self):
        result = walk_forward(
            self.bundle,
            cutoffs=[date(2026, 9, 1)],
            model="backoff",
            horizon_days=30,
        )
        self.assertIn("planned_short", result["excluded"])
        self.assertIn("overdue_no_feedback", result["excluded"])

    def test_backoff_beats_base_rate_on_brier(self):
        result = walk_forward(
            self.bundle,
            cutoffs=[date(2026, 9, 1)],
            model="backoff",
            horizon_days=30,
        )
        baseline = walk_forward(
            self.bundle,
            cutoffs=[date(2026, 9, 1)],
            model="base_rate",
            horizon_days=30,
        )
        self.assertIsNotNone(result["summary"]["brier"])
        self.assertIsNotNone(baseline["summary"]["brier"])
        self.assertLess(result["summary"]["brier"], baseline["summary"]["brier"])


if __name__ == "__main__":
    unittest.main()
