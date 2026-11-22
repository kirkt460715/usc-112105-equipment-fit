"""站长视角的报告与“模型更新会改变哪些建议”的对比。

报告刻意把预测可信度、人员确认、具体配件和保留理由写在同一屏，避免只看
一个风险分数就下结论。
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping

from .data import Bundle
from .labels import STATUS_LABELS, Label
from .model import Prediction
from .planner import Recommendation, StockSnapshot


def _format_interval(prediction: Prediction) -> str:
    if prediction.interval is None:
        return ""
    low, high = prediction.interval
    return f"，区间 {low:.2f}–{high:.2f}"


def render_site_report(
    bundle: Bundle,
    labels: Mapping[str, Label],
    recommendations: Iterable[Recommendation],
    snapshot: StockSnapshot,
    *,
    asof: date,
    model_name: str,
    site: str | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# 换配建议报告（观察时点 {asof.isoformat()}，模型 {model_name}）")
    lines.append("")
    if site:
        lines.append(f"站点筛选：{site}")
        lines.append("")

    grouped: dict[str, list[Recommendation]] = {}
    for recommendation in recommendations:
        if site and recommendation.site != site:
            continue
        grouped.setdefault(recommendation.site, []).append(recommendation)

    for site_name in sorted(grouped):
        lines.append(f"## 站点 {site_name}")
        lines.append("")
        ordered = sorted(grouped[site_name], key=lambda item: item.priority_position)
        for item in ordered:
            lines.append(f"### {item.request}（借用 {item.loan}，第 {item.priority_position} 顺位）")
            lines.append("")
            lines.append(f"- 结论：{item.group} — {item.explanation}")
            if item.device:
                asset = item.device
                if item.parts:
                    asset += " + " + "、".join(item.parts)
                else:
                    asset += "（无配件）"
                if item.handover_on:
                    asset += f"，交接日 {item.handover_on.isoformat()}"
                lines.append(f"- 建议资产：{asset}")
                for part_id in item.parts:
                    part = bundle.part(part_id)
                    lines.append(
                        f"  - 配件 {part_id}（{part.category}，适用 {'、'.join(part.families)}）"
                        f"{'已清点' if part.checked else '经站点清点事件确认'}"
                    )
            lines.append(f"- 人员确认：{'已确认' if item.verified else '未确认'}")
            lines.append(f"- 保护标识：{'是' if item.protected else '否'}")
            lines.append(f"- 剩余使用日：{item.remaining_days}")
            lines.append(f"- 可交回旧件日期：{item.can_return_on.isoformat()}")
            if item.blockers:
                lines.append(f"- 未兑现原因：{'；'.join(item.blockers)}")
            prediction = item.prediction
            if prediction.supported:
                lines.append(
                    f"- 预测可信度：失败风险 {prediction.risk:.2f}"
                    f"（{prediction.level}，历史样本 {prediction.support}，失败 {prediction.failures}"
                    f"{_format_interval(prediction)}）"
                )
                if prediction.level != "L0":
                    scope = "、".join(
                        f"{key}={value}" for key, value in (prediction.sample_scope or {}).items()
                    )
                    lines.append(f"  - 为凑够样本已退让到更粗的配置：{scope}")
            else:
                lines.append(f"- 预测可信度：不足 — {prediction.note}")
            lines.append(f"- 借出标签：{STATUS_LABELS.get(item.label_status, item.label_status)}")
            if item.commitment:
                lines.append(
                    f"- 旧承诺：{item.commitment} 保留至 {item.commitment_until.isoformat()}"
                )
            lines.append("")
    return "\n".join(lines)


def diff_plans(
    before: Mapping[str, Recommendation],
    after: Mapping[str, Recommendation],
) -> list[dict]:
    """比较两次重算（例如模型更新前后）的待选建议。"""

    rows = []
    for request_id in sorted(set(before) | set(after)):
        left = before.get(request_id)
        right = after.get(request_id)
        if left is None or right is None:
            rows.append(
                {
                    "request": request_id,
                    "change": "新增" if left is None else "消失",
                    "before": None if left is None else _asset_summary(left),
                    "after": None if right is None else _asset_summary(right),
                }
            )
            continue
        if _asset_summary(left) == _asset_summary(right):
            continue
        rows.append(
            {
                "request": request_id,
                "change": "建议改变",
                "before": _asset_summary(left),
                "after": _asset_summary(right),
                "before_prediction": _prediction_summary(left),
                "after_prediction": _prediction_summary(right),
            }
        )
    return rows


def _asset_summary(recommendation: Recommendation) -> dict:
    return {
        "outcome": recommendation.outcome,
        "device": recommendation.device,
        "parts": list(recommendation.parts),
        "handover_on": recommendation.handover_on.isoformat() if recommendation.handover_on else None,
        "priority_position": recommendation.priority_position,
    }


def _prediction_summary(recommendation: Recommendation) -> dict:
    prediction = recommendation.prediction
    return {
        "supported": prediction.supported,
        "risk": None if prediction.risk is None else round(prediction.risk, 4),
        "level": prediction.level,
        "support": prediction.support,
    }
