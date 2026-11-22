"""借出时可见的适配特征与可校准的失败风险模型。

特征只来自借出当日已知的资料：站点、用途、所需配件类别、器具尺寸范围、
场地净宽和辅具兼容配件。复核时才补录的尺寸（``reviews.corrected_reach``）
不进入特征，避免把后来的信息写回原来的预测。

模型按“站点/用途/所需配件/尺寸范围/净宽/配件兼容”的精细程度逐级退让，
每一级都要求有足够的历史样本。任何一级都撑不住时返回
``manual_measurement_required``，不给出有把握的风险值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import log, sqrt
from typing import Iterable, Sequence

from .data import Bundle, Device, Loan, Part
from .labels import Label


# 从最细到最粗的特征组合。值都用可哈希的短标签，方便报告直接阅读。
LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("L0", ("site", "use", "needs", "size", "width", "parts")),
    ("L1", ("site", "use", "needs", "size", "width")),
    ("L2", ("site", "use", "needs", "size")),
    ("L3", ("site", "needs", "size")),
    ("L4", ("needs", "size")),
)

DEFAULT_MIN_SUPPORT = 8


@dataclass(frozen=True)
class Features:
    site: str
    use: str
    needs: tuple[str, ...]
    size: str
    width: str
    parts: str
    families: tuple[str, ...]
    matched_devices: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "site": self.site,
            "use": self.use,
            "needs": list(self.needs),
            "size": self.size,
            "width": self.width,
            "parts": self.parts,
            "families": list(self.families),
            "matched_devices": list(self.matched_devices),
        }

    def key(self, fields: Sequence[str]) -> tuple:
        return tuple(getattr(self, name) for name in fields)


def _size_bucket(loan: Loan, devices: Sequence[Device]) -> tuple[str, tuple[Device, ...]]:
    if not devices:
        return "no_device", ()
    in_range = tuple(device for device in devices if device.covers(loan.reach))
    if in_range:
        return "in_range", in_range
    margin = min(
        max(device.size_low - loan.reach, loan.reach - device.size_high) for device in devices
    )
    return ("near_range" if margin <= 5 else "far_from_range"), ()


def _width_bucket(loan: Loan, in_range: Sequence[Device]) -> str:
    if not in_range:
        return "no_sized_device"
    if loan.doorway is None:
        return "unmeasured"
    widths = sorted(device.width for device in in_range)
    if loan.doorway >= widths[-1]:
        return "clear"
    if loan.doorway >= widths[0]:
        return "mixed"
    return "blocked"


def _part_bucket(loan: Loan, families: set[str], parts: Iterable[Part]) -> str:
    if not families:
        return "no_sized_device"
    catalog = list(parts)
    compatible = all(
        any(part.category == category and set(part.families) & families for part in catalog)
        for category in loan.needs
    )
    return "compatible" if compatible else "incompatible"


def extract_features(bundle: Bundle, loan: Loan) -> Features:
    """只使用借出当日已知的资料。"""

    site_devices = [
        device
        for device in bundle.devices.values()
        if device.site == loan.site and loan.use in device.uses
    ]
    size, in_range = _size_bucket(loan, site_devices)
    width = _width_bucket(loan, in_range)
    families = {
        device.family
        for device in in_range
        if loan.doorway is None or loan.doorway >= device.width
    }
    parts = _part_bucket(loan, families, bundle.parts.values())
    return Features(
        site=loan.site,
        use=loan.use,
        needs=tuple(sorted(loan.needs)),
        size=size,
        width=width,
        parts=parts,
        families=tuple(sorted(families)),
        matched_devices=tuple(sorted(device.device for device in in_range)),
    )


def _wilson(failures: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = failures / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Prediction:
    loan: str
    supported: bool
    risk: float | None
    failures: int
    support: int
    level: str | None
    level_key: tuple | None
    sample_scope: dict | None
    interval: tuple[float, float] | None
    features: Features
    status: str
    note: str

    @property
    def manual(self) -> bool:
        return not self.supported

    def as_dict(self) -> dict:
        return {
            "loan": self.loan,
            "supported": self.supported,
            "risk": None if self.risk is None else round(self.risk, 4),
            "failures": self.failures,
            "support": self.support,
            "level": self.level,
            "level_key": list(self.level_key) if self.level_key else None,
            "sample_scope": self.sample_scope,
            "interval": [round(self.interval[0], 4), round(self.interval[1], 4)] if self.interval else None,
            "features": self.features.as_dict(),
            "status": self.status,
            "note": self.note,
        }


class BackoffModel:
    """按样本支撑逐级退让的经验风险模型。

    ``min_support`` 是每一级可被采用的最小样本数。样本不足时不外推；宁可把
    个案送进人工测量，也不排出一个看似确信的风险值。
    """

    name = "backoff"

    def __init__(self, min_support: int = DEFAULT_MIN_SUPPORT):
        if min_support < 1:
            raise ValueError("min_support 至少为 1")
        self.min_support = min_support
        self.levels = LEVELS
        self._counts: list[dict[tuple, list[int]]] = []
        self._totals = [0, 0]
        self.trained_through: date | None = None
        self.training_size = 0

    def fit(self, samples: Sequence[tuple[Loan, Features, bool]], through: date | None = None) -> "BackoffModel":
        self._counts = [{}, {}, {}, {}, {}]
        self._totals = [0, 0]
        self.training_size = 0
        self.trained_through = through
        for _loan, features, positive in samples:
            self._totals[0] += 1
            self._totals[1] += 1 if positive else 0
            self.training_size += 1
            for index, (_name, fields) in enumerate(self.levels):
                key = features.key(fields)
                bucket = self._counts[index].setdefault(key, [0, 0])
                bucket[0] += 1
                bucket[1] += 1 if positive else 0
        return self

    @property
    def base_rate(self) -> float:
        total, failures = self._totals
        if total == 0:
            return 0.5
        return (failures + 0.5) / (total + 1)

    def sample_at(self, index: int, features: Features) -> tuple[int, int]:
        key = features.key(self.levels[index][1])
        total, failures = self._counts[index].get(key, (0, 0))
        return total, failures

    def predict(self, loan_id: str, features: Features) -> Prediction:
        for index, (level_name, fields) in enumerate(self.levels):
            total, failures = self.sample_at(index, features)
            if total >= self.min_support:
                risk = (failures + 0.5) / (total + 1)
                return Prediction(
                    loan=loan_id,
                    supported=True,
                    risk=risk,
                    failures=failures,
                    support=total,
                    level=level_name,
                    level_key=features.key(fields),
                    sample_scope=dict(zip(fields, features.key(fields))),
                    interval=_wilson(failures, total),
                    features=features,
                    status="supported",
                    note=f"{level_name} 有 {total} 个历史样本、{failures} 次适配失败",
                )
        return Prediction(
            loan=loan_id,
            supported=False,
            risk=None,
            failures=0,
            support=0,
            level=None,
            level_key=None,
            sample_scope=None,
            interval=None,
            features=features,
            status="manual_measurement_required",
            note=(
                f"各退让层级都没有达到 {self.min_support} 个历史样本，"
                "不足以给出可信风险，应进入人工测量"
            ),
        )


class BaseRateModel:
    """只使用整体失败率的对照方法。"""

    name = "base_rate"

    def __init__(self, min_support: int = DEFAULT_MIN_SUPPORT):
        self.min_support = min_support
        self.risk = 0.5

    def fit(self, samples: Sequence[tuple[Loan, Features, bool]], through: date | None = None) -> "BaseRateModel":
        total = len(samples)
        failures = sum(1 for _loan, _features, positive in samples if positive)
        self.risk = (failures + 0.5) / (total + 1)
        return self

    def sample_at(self, index: int, features: Features) -> tuple[int, int]:
        return 0, 0

    def predict(self, loan_id: str, features: Features) -> Prediction:
        return Prediction(
            loan=loan_id,
            supported=True,
            risk=self.risk,
            failures=0,
            support=0,
            level="base_rate",
            level_key=(),
            sample_scope={},
            interval=_wilson(0, 0),
            features=features,
            status="supported",
            note="全表平均失败率，不含站点与样本支撑",
        )


class StructuralModel:
    """确定性的结构规则：任一硬条件不满足就判为高风险。

    作为“只看结构、不校准”的对照，不提供概率。
    """

    name = "structural"

    def fit(self, samples: Sequence[tuple[Loan, Features, bool]], through: date | None = None) -> "StructuralModel":
        return self

    def sample_at(self, index: int, features: Features) -> tuple[int, int]:
        return 0, 0

    def predict(self, loan_id: str, features: Features) -> Prediction:
        mismatch = (
            features.size != "in_range"
            or features.width == "blocked"
            or features.parts != "compatible"
        )
        return Prediction(
            loan=loan_id,
            supported=True,
            risk=1.0 if mismatch else 0.0,
            failures=0,
            support=0,
            level="structural",
            level_key=(),
            sample_scope={},
            interval=None,
            features=features,
            status="supported",
            note="结构硬条件不满足即判高风险，未做概率校准",
        )


MODELS = {"backoff": BackoffModel, "base_rate": BaseRateModel, "structural": StructuralModel}


def fit_model(
    bundle: Bundle,
    labels: dict[str, Label],
    asof: date,
    *,
    model: str = "backoff",
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> BackoffModel | BaseRateModel | StructuralModel:
    """只用 ``asof`` 时点已经可见且可训练的标签。"""

    samples = []
    for loan_id, label in labels.items():
        if not label.trainable:
            continue
        if label.observed_on is None or label.observed_on > asof:
            continue
        loan = bundle.loans[loan_id]
        samples.append((loan, extract_features(bundle, loan), label.status == "fit_failure"))
    samples.sort(key=lambda item: (item[0].borrowed_on, item[0].loan))
    factory = MODELS.get(model)
    if factory is None:
        raise ValueError(f"未知模型 {model!r}，可选：{', '.join(sorted(MODELS))}")
    instance = factory(min_support) if model == "backoff" else factory()
    return instance.fit(samples, asof)


# ---- 评估 ---------------------------------------------------------------


def brier(pairs: Sequence[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    return sum((probability - truth) ** 2 for probability, truth in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, bool]], epsilon: float = 1e-6) -> float | None:
    if not pairs:
        return None
    total = 0.0
    for probability, truth in pairs:
        p = min(1 - epsilon, max(epsilon, probability))
        total += -(log(p) if truth else log(1 - p))
    return total / len(pairs)


def reliability(pairs: Sequence[tuple[float, bool]], bins: int = 5) -> list[dict]:
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, truth in pairs:
        index = min(bins - 1, max(0, int(probability * bins)))
        buckets[index].append((probability, truth))
    rows = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            rows.append(
                {
                    "bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                    "n": 0,
                    "mean_predicted": None,
                    "observed_failure_rate": None,
                }
            )
            continue
        rows.append(
            {
                "bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
                "n": len(bucket),
                "mean_predicted": round(sum(p for p, _ in bucket) / len(bucket), 4),
                "observed_failure_rate": round(sum(1 for _, t in bucket if t) / len(bucket), 4),
            }
        )
    return rows


def expected_calibration_error(pairs: Sequence[tuple[float, bool]], bins: int = 5) -> float | None:
    rows = reliability(pairs, bins)
    total = sum(row["n"] for row in rows)
    if total == 0:
        return None
    error = 0.0
    for row in rows:
        if row["n"] == 0:
            continue
        gap = abs(row["mean_predicted"] - row["observed_failure_rate"])
        error += gap * row["n"] / total
    return error


def confusion(
    pairs: Sequence[tuple[float, bool]],
    threshold: float,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 3.0,
) -> dict:
    tp = fp = tn = fn = 0
    for probability, truth in pairs:
        flagged = probability >= threshold
        if flagged and truth:
            tp += 1
        elif flagged and not truth:
            fp += 1
        elif not flagged and truth:
            fn += 1
        else:
            tn += 1
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "cost": round(fp * cost_false_positive + fn * cost_false_negative, 4),
        "false_positive_cost": round(fp * cost_false_positive, 4),
        "false_negative_cost": round(fn * cost_false_negative, 4),
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
    }


def summarise_predictions(
    predictions: Sequence[tuple[Prediction, bool]],
    *,
    threshold: float = 0.5,
    cost_false_positive: float = 1.0,
    cost_false_negative: float = 3.0,
) -> dict:
    supported = [(prediction, truth) for prediction, truth in predictions if prediction.supported]
    pairs = [(prediction.risk or 0.0, truth) for prediction, truth in supported]
    manual = [prediction for prediction, _truth in predictions if not prediction.supported]
    return {
        "n": len(predictions),
        "supported": len(supported),
        "manual": len(manual),
        "coverage": round(len(supported) / len(predictions), 4) if predictions else None,
        "failures": sum(1 for _prediction, truth in supported if truth),
        "brier": None if brier(pairs) is None else round(brier(pairs), 4),
        "log_loss": None if log_loss(pairs) is None else round(log_loss(pairs), 4),
        "ece": None if expected_calibration_error(pairs) is None else round(expected_calibration_error(pairs), 4),
        "confusion": confusion(pairs, threshold, cost_false_positive, cost_false_negative),
        "reliability": reliability(pairs),
        "manual_loans": [prediction.loan for prediction in manual],
    }
