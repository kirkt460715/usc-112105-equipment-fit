"""验证资料读取和标签语义。"""

from datetime import date
import unittest

from exchange_tool.data import DataError, read_bundle
from exchange_tool.labels import chain_groups, label_all, label_loan


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")

    def test_reads_counts(self):
        self.assertEqual(len(self.bundle.loans), 160)
        self.assertEqual(len(self.bundle.requests), 27)
        self.assertEqual(len(self.bundle.devices), 6)

    def test_unknown_loan_raises(self):
        with self.assertRaises(DataError):
            self.bundle.loan("missing")

    def test_chain_groups_join_previous(self):
        groups = chain_groups(self.bundle)
        self.assertEqual(groups["L001"], groups["L000"])
        self.assertNotEqual(groups["L002"], groups["L001"])


class LabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = read_bundle("data")
        cls.grouped_outcomes = {}
        for outcome in cls.bundle.outcomes:
            cls.grouped_outcomes.setdefault(outcome.loan, []).append(outcome)
        cls.grouped_reviews = {}
        for review in cls.bundle.reviews:
            cls.grouped_reviews.setdefault(review.reason, []).append(review)

    def _label(self, loan_id, asof):
        return label_loan(
            self.bundle.loans[loan_id],
            self.grouped_outcomes.get(loan_id, []),
            self.grouped_reviews,
            asof,
        )

    def test_pending_before_due_without_outcome(self):
        label = self._label("L148", date(2026, 11, 22))
        self.assertEqual(label.status, "pending")
        self.assertFalse(label.resolved)

    def test_overdue_without_outcome_is_not_success(self):
        label = self._label("L132", date(2026, 12, 22))
        self.assertEqual(label.status, "overdue_no_feedback")
        self.assertFalse(label.trainable)
        self.assertIsNone(label.positive)

    def test_planned_short_is_not_fit_signal(self):
        label = self._label("L009", date(2026, 12, 22))
        self.assertEqual(label.status, "planned_short")
        self.assertFalse(label.trainable)

    def test_late_confirmation_is_not_success(self):
        # L135 在 2026-11-18 才确认失败，晚于复核期限 2026-11-13，
        # 但它本身是失败，仍可训练；这里检查晚到的确认分支需要构造。
        loan = self.bundle.loans["L135"]
        from exchange_tool.data import Outcome

        late_success = Outcome(
            reason="late",
            loan="L135",
            observed_on=date(2026, 11, 20),
            kind="fit_confirmed",
            evidence="reviewed",
        )
        label = label_loan(loan, [late_success], {}, date(2026, 12, 22))
        self.assertEqual(label.status, "late_unconfirmed")
        self.assertFalse(label.trainable)

    def test_review_replaces_outcome_only_after_its_date(self):
        before = self._label("L019", date(2026, 4, 1))
        after = self._label("L019", date(2026, 5, 1))
        self.assertEqual(before.status, "fit_confirmed")
        self.assertEqual(after.status, "fit_failure")
        self.assertEqual(after.source, "review")

    def test_review_correction_does_not_change_features(self):
        # corrected_reach 只用于更正结论，不写回借出特征；特征层另测。
        label = self._label("L019", date(2026, 5, 1))
        self.assertTrue(label.trainable)
        self.assertEqual(self.bundle.loans["L019"].reach, 104)

    def test_summary_covers_all_loans(self):
        labels = label_all(self.bundle, date(2026, 12, 22))
        self.assertEqual(len(labels), len(self.bundle.loans))


if __name__ == "__main__":
    unittest.main()
