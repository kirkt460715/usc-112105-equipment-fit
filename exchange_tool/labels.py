"""把复核结果、退回理由和日期更正整理成时点标签。

规则要点：

* 只有借出时已经已知的资料可以作为预测输入；``reviews.corrected_reach`` 是
  复核时才补录的尺寸，这里只读取它的更正结论，尺寸本身不进入特征。
* 未到复核期限、超过期限无反馈、短期正常归还、真正的适配失败是四种不同
  状态，不能互相替代。
* 晚于复核期限才出现的“适配确认”不能自动当作成功标签。
* 一个理由可以被后续复核更正；只有 ``observed_on`` 不晚于观察时点的更正
  才可见。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .data import Bundle, Loan, Outcome, Review


STATUS_LABELS = {
    "fit_failure": "适配失败",
    "fit_confirmed": "适配确认",
    "planned_short": "短期正常归还",
    "pending": "未到复核期限",
    "overdue_no_feedback": "超过期限无反馈",
    "late_unconfirmed": "晚到确认（不自动作成功）",
    "unknown": "无法判定",
}


@dataclass(frozen=True)
class Label:
    loan: str
    status: str
    observed_on: date | None
    source: str
    reason: str | None
    detail: str

    @property
    def trainable(self) -> bool:
        """是否可作为适配成功/失败的训练标签。"""

        return self.status in {"fit_confirmed", "fit_failure"}

    @property
    def positive(self) -> bool | None:
        if self.status == "fit_failure":
            return True
        if self.status == "fit_confirmed":
            return False
        return None

    @property
    def resolved(self) -> bool:
        return self.status in {"fit_confirmed", "fit_failure", "planned_short"}


def _outcomes_by_loan(outcomes: Iterable[Outcome]) -> dict[str, list[Outcome]]:
    grouped: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.loan, []).append(outcome)
    return grouped


def _reviews_by_reason(reviews: Iterable[Review]) -> dict[str, list[Review]]:
    grouped: dict[str, list[Review]] = {}
    for review in reviews:
        grouped.setdefault(review.reason, []).append(review)
    return grouped


def label_loan(
    loan: Loan,
    outcomes: list[Outcome],
    reviews: dict[str, list[Review]],
    asof: date,
) -> Label:
    """在 ``asof`` 时点给一笔借用定状态。

    只使用 ``observed_on <= asof`` 的事实。没有可见结果时，以复核期限区分
    “未到期限”和“超过期限无反馈”。
    """

    visible = [item for item in outcomes if item.observed_on <= asof]
    if not visible:
        if asof <= loan.due:
            return Label(
                loan.loan,
                "pending",
                None,
                "none",
                None,
                f"{asof.isoformat()} 尚未到复核期限 {loan.due.isoformat()}，不能当作成功或失败",
            )
        return Label(
            loan.loan,
            "overdue_no_feedback",
            None,
            "none",
            None,
            f"复核期限 {loan.due.isoformat()} 已过且没有可见反馈，单独计为无反馈",
        )

    outcome = max(visible, key=lambda item: (item.observed_on, item.reason))
    reason = outcome.reason
    kind = outcome.kind
    observed_on = outcome.observed_on
    source = "outcome"

    revisions = [rev for rev in reviews.get(reason, []) if rev.observed_on <= asof]
    if revisions:
        review = max(revisions, key=lambda item: (item.observed_on, item.revision))
        kind = review.replacement
        observed_on = review.observed_on
        source = "review"

    if kind == "planned_short":
        return Label(
            loan.loan,
            "planned_short",
            observed_on,
            source,
            reason,
            "原本只借几天并按计划正常归还，不作为适配失败或成功标签",
        )
    if kind == "fit_failure":
        return Label(
            loan.loan,
            "fit_failure",
            observed_on,
            source,
            reason,
            "完成复核或退回理由判定为适配失败",
        )
    if kind == "fit_confirmed":
        if observed_on > loan.due:
            return Label(
                loan.loan,
                "late_unconfirmed",
                observed_on,
                source,
                reason,
                f"确认发生在复核期限 {loan.due.isoformat()} 之后，不能自动当作成功标签",
            )
        return Label(
            loan.loan,
            "fit_confirmed",
            observed_on,
            source,
            reason,
            "复核期限内完成适配确认",
        )
    return Label(
        loan.loan,
        "unknown",
        observed_on,
        source,
        reason,
        f"结果类型 {kind!r} 不在约定内",
    )


def label_all(bundle: Bundle, asof: date) -> dict[str, Label]:
    grouped_outcomes = _outcomes_by_loan(bundle.outcomes)
    grouped_reviews = _reviews_by_reason(bundle.reviews)
    return {
        loan_id: label_loan(loan, grouped_outcomes.get(loan_id, []), grouped_reviews, asof)
        for loan_id, loan in bundle.loans.items()
    }


def label_summary(labels: dict[str, Label]) -> dict[str, int]:
    counts = {status: 0 for status in STATUS_LABELS}
    for label in labels.values():
        counts[label.status] = counts.get(label.status, 0) + 1
    return counts


def chain_groups(bundle: Bundle) -> dict[str, int]:
    """按 ``previous`` 关系把同一人的续借/换配链合并成组。

    同一次借用的调换和续借必须落在训练或回测的同一侧，因此回测按组切分。
    """

    parent: dict[str, str] = {loan_id: loan_id for loan_id in bundle.loans}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for loan_id, loan in bundle.loans.items():
        if loan.previous is not None:
            union(loan_id, loan.previous)

    roots = {loan_id: find(loan_id) for loan_id in bundle.loans}
    index: dict[str, int] = {}
    for loan_id in sorted(roots):
        index.setdefault(roots[loan_id], len(index))
    return {loan_id: index[root] for loan_id, root in roots.items()}
