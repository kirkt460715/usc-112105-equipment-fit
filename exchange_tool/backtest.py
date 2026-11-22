"""按时点与站点回测预测方法。

回测采用前向验证：每个观察时点只用此前已经可见的标签训练，再预测时点之后
借出的个案，最后对照后来完成复核的结论。切分按 ``previous`` 形成的借用链
整组进行，保证同一次借用的调换和续借不会一边在训练、一边在回测。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

from .data import Bundle
from .labels import Label, chain_groups, label_all
from .model import (
    Prediction,
    extract_features,
    fit_model,
    summarise_predictions,
)


@dataclass
class Fold:
    asof: date
    train_loans: int
    test_loans: int
    dropped_chain_overlap: int


def month_cutoffs(start: date, end: date) -> list[date]:
    cutoffs = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        cutoffs.append(cursor - timedelta(days=1))
        cursor = date(cursor.year + (cursor.month == 12), (cursor.month % 12) + 1, 1)
    return [cutoff for cutoff in cutoffs if start <= cutoff <= end]


def default_cutoffs(bundle: Bundle, step_days: int = 30, minimum_training: int = 30) -> list[date]:
    borrowed = sorted(loan.borrowed_on for loan in bundle.loans.values())
    if not borrowed:
        return []
    start = borrowed[0] + timedelta(days=minimum_training)
    end = max(borrowed)
    cutoffs = []
    cursor = start
    while cursor < end:
        cutoffs.append(cursor)
        cursor += timedelta(days=step_days)
    return cutoffs


def _final_labels(bundle: Bundle, asof: date) -> dict[str, Label]:
    return label_all(bundle, asof)


def walk_forward(
    bundle: Bundle,
    *,
    cutoffs: Sequence[date],
    model: str = "backoff",
    min_support: int = 8,
    sites: Sequence[str] | None = None,
    horizon_days: int | None = None,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 3.0,
    threshold: float = 0.5,
    final_asof: date | None = None,
) -> dict:
    """对每个观察时点训练并预测之后借出的个案。

    ``horizon_days`` 控制预测距离：None 表示预测时点之后的全部个案。
    预测标签以 ``final_asof``（默认全部资料可见）为准，且仍然遵守
    “晚到确认不作成功”的规则。
    """

    if not cutoffs:
        return {"model": model, "folds": [], "summary": None}
    final_asof = final_asof or max(
        [loan.due for loan in bundle.loans.values()]
        + [outcome.observed_on for outcome in bundle.outcomes]
        + [review.observed_on for review in bundle.reviews]
        + [event.observed_on for event in bundle.stock_events]
    )
    final_labels = _final_labels(bundle, final_asof)
    chains = chain_groups(bundle)
    site_filter = set(sites) if sites else None

    folds = []
    all_predictions: list[tuple[Prediction, bool]] = []
    evaluated: list[dict] = []
    tested: set[str] = set()
    for asof in sorted(cutoffs):
        labels = label_all(bundle, asof)
        training_ids = {
            loan_id
            for loan_id, label in labels.items()
            if label.trainable and label.observed_on is not None and label.observed_on <= asof
        }
        training_chains = {chains[loan_id] for loan_id in training_ids}
        cutoff = asof
        horizon_end = cutoff + timedelta(days=horizon_days) if horizon_days else None
        test_ids = []
        dropped = 0
        for loan_id, loan in bundle.loans.items():
            if loan_id in tested:
                continue
            if not (cutoff < loan.borrowed_on <= (horizon_end or cutoff + timedelta(days=3650))):
                continue
            if site_filter is not None and loan.site not in site_filter:
                continue
            if not final_labels[loan_id].trainable:
                continue
            if chains[loan_id] in training_chains:
                dropped += 1
                continue
            test_ids.append(loan_id)
        tested.update(test_ids)
        model_instance = fit_model(bundle, labels, asof, model=model, min_support=min_support)
        for loan_id in sorted(test_ids, key=lambda item: (bundle.loans[item].borrowed_on, item)):
            loan = bundle.loans[loan_id]
            features = extract_features(bundle, loan)
            prediction = model_instance.predict(loan_id, features)
            truth = final_labels[loan_id].status == "fit_failure"
            all_predictions.append((prediction, truth))
            evaluated.append(
                {
                    "loan": loan_id,
                    "site": loan.site,
                    "borrowed_on": loan.borrowed_on.isoformat(),
                    "asof": asof.isoformat(),
                    "status": final_labels[loan_id].status,
                    "truth_failure": truth,
                    "risk": None if prediction.risk is None else round(prediction.risk, 4),
                    "supported": prediction.supported,
                    "level": prediction.level,
                    "support": prediction.support,
                }
            )
        folds.append(
            Fold(
                asof=asof,
                train_loans=len(training_ids),
                test_loans=len(test_ids),
                dropped_chain_overlap=dropped,
            )
        )

    summary = summarise_predictions(
        all_predictions,
        threshold=threshold,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
    )
    site_rows = []
    for site in sorted({row["site"] for row in evaluated}):
        subset = [
            (prediction, truth)
            for (prediction, truth), row in zip(all_predictions, evaluated)
            if row["site"] == site
        ]
        site_summary = summarise_predictions(
            subset,
            threshold=threshold,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        )
        site_rows.append({"site": site, **site_summary})
    fold_rows = []
    for fold in folds:
        subset = [
            (prediction, truth)
            for (prediction, truth), item in zip(all_predictions, evaluated)
            if item["asof"] == fold.asof.isoformat()
        ]
        fold_summary = summarise_predictions(
            subset,
            threshold=threshold,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
        )
        fold_rows.append(
            {
                "asof": fold.asof.isoformat(),
                "train_loans": fold.train_loans,
                "test_loans": fold.test_loans,
                "dropped_chain_overlap": fold.dropped_chain_overlap,
                "brier": fold_summary["brier"],
                "ece": fold_summary["ece"],
                "coverage": fold_summary["coverage"],
                "cost": fold_summary["confusion"]["cost"],
            }
        )
    return {
        "model": model,
        "min_support": min_support,
        "threshold": threshold,
        "costs": {"false_positive": cost_false_positive, "false_negative": cost_false_negative},
        "folds": fold_rows,
        "summary": summary,
        "sites": site_rows,
        "cases": evaluated,
        "excluded": _excluded_summary(bundle, final_labels, site_filter),
    }


def _excluded_summary(
    bundle: Bundle, labels: dict[str, Label], sites: set[str] | None
) -> dict:
    """未进入评估的状态单独计数，便于看出有多少个案被排除在外及原因。"""

    counts: dict[str, int] = {}
    for loan_id, label in labels.items():
        if label.trainable:
            continue
        if sites is not None and bundle.loans[loan_id].site not in sites:
            continue
        counts[label.status] = counts.get(label.status, 0) + 1
    return counts


def compare_models(
    bundle: Bundle,
    *,
    cutoffs: Sequence[date],
    models: Sequence[str] = ("backoff", "base_rate", "structural"),
    min_support: int = 8,
    sites: Sequence[str] | None = None,
    horizon_days: int | None = None,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 3.0,
    threshold: float = 0.5,
) -> dict:
    results = {}
    for model in models:
        results[model] = walk_forward(
            bundle,
            cutoffs=cutoffs,
            model=model,
            min_support=min_support,
            sites=sites,
            horizon_days=horizon_days,
            cost_false_positive=cost_false_positive,
            cost_false_negative=cost_false_negative,
            threshold=threshold,
        )
    return results
