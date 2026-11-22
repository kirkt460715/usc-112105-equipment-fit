"""站点可反复更新的分析入口。

只依赖标准库。每次运行都从资料目录重新读取，因此站点追加新的复核、退回、
原因更正和到场确认后，重算即可得到新的候选顺位与可用库存。

用法示例::

    python3 -m exchange_tool report --asof 2026-12-22
    python3 -m exchange_tool predict --asof 2026-12-22 --site bridge
    python3 -m exchange_tool backtest --step 30 --horizon 30
    python3 -m exchange_tool backtest --model backoff --compare base_rate structural
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import sys

from .backtest import compare_models, default_cutoffs, walk_forward
from .data import DataError, read_bundle
from .labels import STATUS_LABELS, label_all, label_summary
from .model import MODELS, DEFAULT_MIN_SUPPORT, extract_features, fit_model
from .planner import compare_schedules, plan_exchanges, risk_only_plan, stock_snapshot
from .report import diff_plans, render_site_report


def _date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"不是有效日期：{value!r}") from error


def _bundle(args):
    return read_bundle(args.directory)


def cmd_inspect(args) -> int:
    bundle = _bundle(args)
    payload = {
        "directory": str(bundle.directory),
        "counts": {
            "loans": len(bundle.loans),
            "outcomes": len(bundle.outcomes),
            "reviews": len(bundle.reviews),
            "devices": len(bundle.devices),
            "parts": len(bundle.parts),
            "requests": len(bundle.requests),
            "reservations": len(bundle.reservations),
            "stock_events": len(bundle.stock_events),
        },
        "warnings": bundle.warnings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _request_predictions(bundle, asof: date, model: str, min_support: int) -> tuple[dict, dict]:
    labels = label_all(bundle, asof)
    fitted = fit_model(bundle, labels, asof, model=model, min_support=min_support)
    predictions = {
        request_id: fitted.predict(request.loan, extract_features(bundle, bundle.loan(request.loan)))
        for request_id, request in bundle.requests.items()
    }
    return labels, predictions


def cmd_predict(args) -> int:
    bundle = _bundle(args)
    labels, predictions = _request_predictions(bundle, args.asof, args.model, args.min_support)
    rows = []
    for request_id in sorted(predictions, key=lambda item: bundle.request(item).loan):
        request = bundle.request(request_id)
        loan = bundle.loan(request.loan)
        if args.site and loan.site != args.site:
            continue
        prediction = predictions[request_id]
        rows.append(
            {
                "request": request_id,
                "loan": request.loan,
                "site": loan.site,
                "label": labels[loan.loan].status,
                "label_detail": labels[loan.loan].detail,
                "prediction": prediction.as_dict(),
            }
        )
    print(
        json.dumps(
            {
                "asof": args.asof.isoformat(),
                "model": args.model,
                "min_support": args.min_support,
                "labels": label_summary(labels),
                "request_predictions": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_plan(args) -> int:
    bundle = _bundle(args)
    labels, predictions = _request_predictions(bundle, args.asof, args.model, args.min_support)
    plan = plan_exchanges(bundle, predictions, labels, args.asof, model_name=args.model)
    if args.site:
        plan = {
            request_id: recommendation
            for request_id, recommendation in plan.items()
            if recommendation.site == args.site
        }
    risky = risk_only_plan(bundle, predictions, labels, args.asof)
    comparison = compare_schedules(plan, risky, bundle)
    print(
        json.dumps(
            {
                "asof": args.asof.isoformat(),
                "model": args.model,
                "min_support": args.min_support,
                "stock": stock_snapshot(bundle, args.asof).as_dict(),
                "recommendations": [item.as_dict() for item in plan.values()],
                "risk_only_comparison": comparison,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_backtest(args) -> int:
    bundle = _bundle(args)
    cutoffs = (
        [_date_arg(value) for value in args.cutoffs]
        if args.cutoffs
        else default_cutoffs(bundle, step_days=args.step)
    )
    common = dict(
        cutoffs=cutoffs,
        min_support=args.min_support,
        sites=args.site,
        horizon_days=args.horizon,
        cost_false_positive=args.false_positive_cost,
        cost_false_negative=args.false_negative_cost,
        threshold=args.threshold,
    )
    if args.compare:
        results = compare_models(
            bundle, models=[args.model, *args.compare], **common
        )
        payload = {"cutoffs": [item.isoformat() for item in cutoffs], "models": results}
    else:
        payload = walk_forward(bundle, model=args.model, **common)
        payload["cutoffs"] = [item.isoformat() for item in cutoffs]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_report(args) -> int:
    """站长视角：每个建议的依据、预测可信度、人员确认与具体配件。"""

    bundle = _bundle(args)
    labels, predictions = _request_predictions(bundle, args.asof, args.model, args.min_support)
    plan = plan_exchanges(bundle, predictions, labels, args.asof, model_name=args.model)
    snapshot = stock_snapshot(bundle, args.asof)

    body = render_site_report(
        bundle,
        labels,
        plan.values(),
        snapshot,
        asof=args.asof,
        model_name=args.model,
        site=args.site,
    )
    lines = body.splitlines()
    lines.extend(["", "## 标签分布", ""])
    for status, count in sorted(label_summary(labels).items()):
        lines.append(f"- {STATUS_LABELS.get(status, status)}：{count}")
    lines.extend(["", "## 可用库存", ""])
    lines.append(f"- 可再用配件：{'、'.join(sorted(snapshot.reusable_parts)) or '无'}")
    for asset, note in sorted(snapshot.notes.items()):
        lines.append(f"- {asset}：{note}")
    print("\n".join(lines))
    return 0


def cmd_diff(args) -> int:
    """比较两个观察时点/模型的换配建议，说明模型更新本身改变了什么。"""

    bundle = _bundle(args)
    before_labels, before_predictions = _request_predictions(
        bundle, args.before, args.before_model, args.min_support
    )
    after_labels, after_predictions = _request_predictions(
        bundle, args.after, args.after_model, args.min_support
    )
    before = plan_exchanges(
        bundle, before_predictions, before_labels, args.before, model_name=args.before_model
    )
    after = plan_exchanges(
        bundle, after_predictions, after_labels, args.after, model_name=args.after_model
    )
    rows = diff_plans(before, after)
    print(
        json.dumps(
            {
                "before": {"asof": args.before.isoformat(), "model": args.before_model},
                "after": {"asof": args.after.isoformat(), "model": args.after_model},
                "changes": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="辅具借用换配：站点可反复更新的分析工具")
    parser.add_argument("--directory", default="data", help="资料目录，默认 data")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_parser = sub.add_parser("inspect", help="读取资料并显示表计数")
    inspect_parser.set_defaults(func=cmd_inspect)

    predict_parser = sub.add_parser("predict", help="给出待换配个案的预测与标签")
    predict_parser.add_argument("--asof", type=_date_arg, required=True)
    predict_parser.add_argument("--model", choices=sorted(MODELS), default="backoff")
    predict_parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    predict_parser.add_argument("--site")
    predict_parser.set_defaults(func=cmd_predict)

    plan_parser = sub.add_parser("plan", help="给出可兑现的换配安排，并与单按风险排队对比")
    plan_parser.add_argument("--asof", type=_date_arg, required=True)
    plan_parser.add_argument("--model", choices=sorted(MODELS), default="backoff")
    plan_parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    plan_parser.add_argument("--site")
    plan_parser.set_defaults(func=cmd_plan)

    backtest_parser = sub.add_parser("backtest", help="按时点/站点做前向回测")
    backtest_parser.add_argument("--model", choices=sorted(MODELS), default="backoff")
    backtest_parser.add_argument("--compare", nargs="*", choices=sorted(MODELS), default=None)
    backtest_parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    backtest_parser.add_argument("--step", type=int, default=30, help="自动切点的间隔天数")
    backtest_parser.add_argument("--cutoffs", nargs="*", help="指定观察时点（YYYY-MM-DD）")
    backtest_parser.add_argument("--horizon", type=int, default=30, help="预测距离（天），默认 30")
    backtest_parser.add_argument("--site", action="append", help="限定站点，可重复")
    backtest_parser.add_argument("--threshold", type=float, default=0.5)
    backtest_parser.add_argument("--false-positive-cost", type=float, default=1.0)
    backtest_parser.add_argument("--false-negative-cost", type=float, default=3.0)
    backtest_parser.set_defaults(func=cmd_backtest)

    report_parser = sub.add_parser("report", help="站长视角的换配建议报告")
    report_parser.add_argument("--asof", type=_date_arg, required=True)
    report_parser.add_argument("--model", choices=sorted(MODELS), default="backoff")
    report_parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    report_parser.add_argument("--site")
    report_parser.set_defaults(func=cmd_report)

    diff_parser = sub.add_parser("diff", help="比较前后两次重算的换配建议")
    diff_parser.add_argument("--before", type=_date_arg, required=True)
    diff_parser.add_argument("--after", type=_date_arg, required=True)
    diff_parser.add_argument("--before-model", choices=sorted(MODELS), default="backoff")
    diff_parser.add_argument("--after-model", choices=sorted(MODELS), default="backoff")
    diff_parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    diff_parser.set_defaults(func=cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DataError as error:
        print(f"资料错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
