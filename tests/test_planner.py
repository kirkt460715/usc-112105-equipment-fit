"""验证可兑现的换配安排与库存规则。"""

from datetime import date
import unittest

from exchange_tool.data import read_bundle
from exchange_tool.labels import label_all
from exchange_tool.model import extract_features, fit_model
from exchange_tool.planner import (
    compare_schedules,
    plan_exchanges,
    risk_only_plan,
    stock_snapshot,
)


class StockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")

    def test_unchecked_part_not_reusable(self):
        snapshot = stock_snapshot(self.bundle, date(2026, 12, 19))
        self.assertNotIn("K2", snapshot.reusable_parts)
        self.assertIn("尚未完成清点", snapshot.notes["K2"])

    def test_checked_part_becomes_reusable_only_after_check(self):
        before = stock_snapshot(self.bundle, date(2026, 12, 21))
        after = stock_snapshot(self.bundle, date(2026, 12, 22))
        self.assertNotIn("K2", before.reusable_parts)
        self.assertIn("K2", after.reusable_parts)

    def test_collected_device_leaves_stock(self):
        snapshot = stock_snapshot(self.bundle, date(2026, 12, 22))
        self.assertNotIn("D2", snapshot.available_devices)


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")
        cls.asof = date(2026, 12, 22)
        cls.labels = label_all(cls.bundle, cls.asof)
        cls.model = fit_model(cls.bundle, cls.labels, cls.asof, model="backoff", min_support=8)
        cls.predictions = {
            request_id: cls.model.predict(
                request.loan, extract_features(cls.bundle, cls.bundle.loan(request.loan))
            )
            for request_id, request in cls.bundle.requests.items()
        }
        cls.plan = plan_exchanges(cls.bundle, cls.predictions, cls.labels, cls.asof)
        cls.risk_only = risk_only_plan(cls.bundle, cls.predictions, cls.labels, cls.asof)

    def test_commitments_survive_high_risk(self):
        # q140 的预测风险很高，但它有未过期的已签收保留，必须继续保留。
        self.assertEqual(self.plan["q140"].outcome, "existing_commitment")
        self.assertEqual(self.plan["q140"].device, "D3")
        self.assertIn("G1", self.plan["q140"].parts)
        self.assertEqual(self.plan["q133"].outcome, "existing_commitment")

    def test_commitment_on_collected_device_is_kept(self):
        # D2 已按 r1 领取，但仍不能分给别的个案。
        self.assertEqual(self.plan["q133"].device, "D2")
        self.assertNotEqual(
            [item.device for item in self.plan.values() if item.request != "q133"],
            ["D2"],
        )

    def test_matched_plan_has_all_parts_available(self):
        matched = [item for item in self.plan.values() if item.outcome == "matched"]
        self.assertTrue(matched)
        snapshot = stock_snapshot(self.bundle, self.asof)
        for item in matched:
            self.assertIn(item.device, snapshot.available_devices)
            for part in item.parts:
                self.assertIn(part, snapshot.reusable_parts)

    def test_manual_configs_do_not_receive_assets(self):
        for request_id, prediction in self.predictions.items():
            item = self.plan[request_id]
            if item.commitment:
                # 旧承诺优先于样本支撑；已答应领取的组合继续保留。
                self.assertEqual(item.outcome, "existing_commitment")
                continue
            if prediction.manual:
                self.assertEqual(item.outcome, "manual_measurement")
                self.assertIsNone(item.device)

    def test_conflict_categories_are_distinguished(self):
        outcomes = {item.outcome for item in self.plan.values()}
        self.assertIn("no_feasible_date", outcomes)
        self.assertIn("higher_priority", outcomes)
        self.assertIn("no_compatible", outcomes)

    def test_priority_ordering_uses_protection(self):
        self.assertTrue(self.plan["q147"].protected)
        # q147 有保护标识且在 bridge 内拿到唯一可兑现的 D1。
        self.assertEqual(self.plan["q147"].outcome, "matched")
        self.assertLess(self.plan["q147"].priority_position, self.plan["q150"].priority_position)

    def test_risk_only_comparison_changes_allocation(self):
        rows = {row["request"]: row for row in compare_schedules(self.plan, self.risk_only, self.bundle)}
        differences = [
            request for request, row in rows.items() if row["difference"] == "资源去向不同"
        ]
        self.assertTrue(differences)

    def test_handover_date_never_precedes_return_date(self):
        for item in self.plan.values():
            if item.handover_on is not None:
                self.assertGreaterEqual(item.handover_on, item.can_return_on)


if __name__ == "__main__":
    unittest.main()
