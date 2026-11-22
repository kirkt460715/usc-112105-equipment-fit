"""验证站长报告与前后重算对比。"""

from datetime import date
import json
from pathlib import Path
import subprocess
import sys
import unittest

from exchange_tool.data import read_bundle
from exchange_tool.labels import label_all
from exchange_tool.model import extract_features, fit_model
from exchange_tool.planner import plan_exchanges, stock_snapshot
from exchange_tool.report import diff_plans, render_site_report

ROOT = Path(__file__).resolve().parents[1]


def build_plan(bundle, asof, model="backoff", min_support=8):
    labels = label_all(bundle, asof)
    fitted = fit_model(bundle, labels, asof, model=model, min_support=min_support)
    predictions = {
        request_id: fitted.predict(
            request.loan, extract_features(bundle, bundle.loan(request.loan))
        )
        for request_id, request in bundle.requests.items()
    }
    return labels, plan_exchanges(bundle, predictions, labels, asof)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle(ROOT / "data")
        cls.asof = date(2026, 12, 22)
        cls.labels, cls.plan = build_plan(cls.bundle, cls.asof)

    def test_report_shows_prediction_support_and_parts(self):
        text = render_site_report(
            self.bundle,
            self.labels,
            self.plan.values(),
            stock_snapshot(self.bundle, self.asof),
            asof=self.asof,
            model_name="backoff",
            site="bridge",
        )
        self.assertIn("## 站点 bridge", text)
        self.assertNotIn("## 站点 lake", text)
        self.assertIn("预测可信度", text)
        self.assertIn("人员确认", text)
        self.assertIn("配件 K2", text)
        self.assertIn("q147", text)

    def test_report_keeps_commitments_with_reason(self):
        text = render_site_report(
            self.bundle,
            self.labels,
            self.plan.values(),
            stock_snapshot(self.bundle, self.asof),
            asof=self.asof,
            model_name="backoff",
            site="forest",
        )
        # 旧承诺的解释必须写进报告：为什么高风险的旧承诺继续保留。
        self.assertIn("旧承诺", text)
        self.assertIn("保留至", text)
        self.assertIn("q140", text)

    def test_report_marks_manual_measurement(self):
        text = render_site_report(
            self.bundle,
            self.labels,
            self.plan.values(),
            stock_snapshot(self.bundle, self.asof),
            asof=self.asof,
            model_name="backoff",
        )
        self.assertIn("人工测量", text)

    def test_diff_detects_later_stock_update(self):
        _labels_before, plan_before = build_plan(self.bundle, date(2026, 12, 20))
        _labels_after, plan_after = build_plan(self.bundle, date(2026, 12, 22))
        rows = {row["request"]: row for row in diff_plans(plan_before, plan_after)}
        self.assertIn("q147", rows)
        self.assertEqual(rows["q147"]["after"]["device"], "D1")
        # 旧承诺在两次重算中都保留。
        self.assertNotIn("q140", rows)
        self.assertNotIn("q133", rows)

    def test_diff_reports_model_switch(self):
        _labels, plan_after = build_plan(self.bundle, self.asof, model="base_rate")
        rows = diff_plans(self.plan, plan_after)
        self.assertTrue(rows)


class CliTests(unittest.TestCase):
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "exchange_tool", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    def test_predict_json(self):
        result = self._run("predict", "--asof", "2026-12-22", "--site", "bridge")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual({row["site"] for row in payload["request_predictions"]}, {"bridge"})

    def test_plan_and_report_exit_cleanly(self):
        for command in ("plan", "report"):
            result = self._run(command, "--asof", "2026-12-22")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)

    def test_bad_asof_is_rejected(self):
        result = self._run("plan", "--asof", "not-a-date")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
