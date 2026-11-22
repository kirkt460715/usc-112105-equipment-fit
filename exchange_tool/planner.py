"""可兑现的换配安排。

规则要点：

* 实际调换只接受资料里由工作人员认可的兼容范围：器具的站点、合规用途、
  可调尺寸和通过净宽；配件的类别与适用器具系列，并且必须已确认可再次入库。
* 可调主件换了尺寸能服务某人，但配套的支撑件可能正被另一个保留占用。主件
  和每个所需配件必须在同一个交接日都可兑现。
* 只有已经核实退回并通过清点的配件才能进入新的保留。已签收（collected）
  未再入库的资产不可用。
* 已答应领取、尚未过期或改签的保留，优先保留其主件和共同配件。
* 优先级依次看：旧承诺、保护标识、失败证据、人员确认、剩余使用日、可信风险
  和可交接日期。安排不可兑现时，区分“匹配不到”“交接时间冲突”和“有更高
  优先级的安排”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .data import Bundle, Device, Part, Request, Reservation
from .labels import Label
from .model import Prediction


GROUPS = {
    "existing_commitment": "旧承诺",
    "matched": "可兑现安排",
    "no_compatible": "匹配不到",
    "width_unverified": "待人工测量",
    "manual_measurement": "待人工测量",
    "no_feasible_date": "交接时间冲突",
    "handover_conflict": "交接时间冲突",
    "existing_commitment_hold": "交接时间冲突",
    "higher_priority": "更高优先级",
}

OUTCOME_LABELS = {
    "existing_commitment": "旧承诺保留",
    "matched": "匹配到可兑现安排",
    "no_compatible": "没有兼容的器具与配件组合",
    "width_unverified": "净宽未测量，无法认可实际调换",
    "manual_measurement": "样本不足，需人工测量",
    "no_feasible_date": "可交回日期晚于全部交接日期",
    "handover_conflict": "兼容组合在交接日不可兑现",
    "existing_commitment_hold": "资产由未到期旧承诺占用",
    "higher_priority": "资源已保留给旧承诺或更高优先级安排",
}


@dataclass(frozen=True)
class StockSnapshot:
    """某个时点可用于新保留的资产。"""

    reusable_parts: frozenset[str]
    available_devices: frozenset[str]
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "reusable_parts": sorted(self.reusable_parts),
            "available_devices": sorted(self.available_devices),
            "notes": self.notes,
        }


def _last_stock_event(bundle: Bundle, asset: str, asof: date):
    events = [
        event
        for event in bundle.stock_events
        if event.asset == asset and event.observed_on <= asof
    ]
    if not events:
        return None
    return max(events, key=lambda item: (item.observed_on, item.event))


def _in_stock(bundle: Bundle, asset: str, asof: date) -> bool:
    last = _last_stock_event(bundle, asset, asof)
    return last is None or last.kind in {"returned", "checked"}


def _part_checked(bundle: Bundle, part: Part, asof: date) -> bool:
    if part.checked:
        return True
    return any(
        event.asset == part.part
        and event.kind == "checked"
        and event.observed_on <= asof
        for event in bundle.stock_events
    )


def stock_snapshot(bundle: Bundle, asof: date) -> StockSnapshot:
    reusable_parts = set()
    notes = {}
    for part in bundle.parts.values():
        checked = _part_checked(bundle, part, asof)
        in_stock = _in_stock(bundle, part.part, asof)
        if checked and in_stock:
            reusable_parts.add(part.part)
        elif not checked:
            notes[part.part] = "尚未完成清点确认，不可用于新保留"
        else:
            notes[part.part] = "已被领取且未再入库，不可用于新保留"
    available_devices = {
        device.device
        for device in bundle.devices.values()
        if _in_stock(bundle, device.device, asof)
    }
    for device in bundle.devices.values():
        if device.device not in available_devices:
            notes[device.device] = "已被领取且未再入库，不可用于新安排"
    return StockSnapshot(frozenset(reusable_parts), frozenset(available_devices), notes)


def _active_reservations(bundle: Bundle, asof: date) -> list[Reservation]:
    return [
        reservation
        for reservation in bundle.reservations
        if reservation.state == "accepted" and reservation.until_on >= asof
    ]


def _commitment_for(bundle: Bundle, request_id: str, asof: date) -> Reservation | None:
    candidates = [
        reservation
        for reservation in _active_reservations(bundle, asof)
        if reservation.request == request_id
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.until_on, item.reservation))


def _asset_hold_until(bundle: Bundle, asset: str, asof: date) -> tuple[Reservation, date] | None:
    latest = None
    for reservation in _active_reservations(bundle, asof):
        if reservation.device == asset or asset in reservation.parts:
            if latest is None or reservation.until_on > latest[1]:
                latest = (reservation, reservation.until_on)
    return latest


def _compatible_parts_for(
    bundle: Bundle, device: Device, categories: Sequence[str], snapshot: StockSnapshot
) -> dict[str, tuple[str, ...]]:
    """每个所需类别下，与器具系列兼容且已清点可用的配件。"""

    result = {}
    for category in categories:
        options = tuple(
            sorted(
                part.part
                for part in bundle.parts.values()
                if part.category == category
                and device.family in part.families
                and part.part in snapshot.reusable_parts
            )
        )
        result[category] = options
    return result


@dataclass(frozen=True)
class Option:
    device: Device
    parts_by_category: dict[str, tuple[str, ...]]
    feasible_dates: tuple[date, ...]

    def complete(self) -> bool:
        return bool(self.parts_by_category) and all(
            options for options in self.parts_by_category.values()
        )


def _feasible_dates(request: Request, asof: date) -> tuple[date, ...]:
    return tuple(
        sorted(
            day
            for day in request.exchange_dates
            if day >= asof and day >= request.can_return_on
        )
    )


def _options_for(
    bundle: Bundle, request: Request, snapshot: StockSnapshot, asof: date
) -> tuple[list[Option], str | None]:
    loan = bundle.loan(request.loan)
    categories = tuple(sorted(set(loan.needs)))
    dates = _feasible_dates(request, asof)
    options: list[Option] = []
    width_blocked = False
    for device in sorted(bundle.devices.values(), key=lambda item: item.device):
        if device.site != loan.site or loan.use not in device.uses:
            continue
        if not device.covers(loan.reach):
            continue
        if loan.doorway is None:
            width_blocked = True
            continue
        if loan.doorway < device.width:
            continue
        options.append(
            Option(device, _compatible_parts_for(bundle, device, categories, snapshot), dates)
        )
    if not options and width_blocked:
        return [], "width_unverified"
    return options, None


@dataclass(frozen=True)
class Recommendation:
    request: str
    loan: str
    site: str
    priority_position: int
    priority_key: tuple
    prediction: Prediction
    label_status: str
    verified: bool
    protected: bool
    remaining_days: int
    can_return_on: date
    outcome: str
    group: str
    device: str | None
    parts: tuple[str, ...]
    handover_on: date | None
    explanation: str
    blockers: tuple[str, ...]
    commitment: str | None
    commitment_until: date | None

    def as_dict(self) -> dict:
        return {
            "request": self.request,
            "loan": self.loan,
            "site": self.site,
            "priority_position": self.priority_position,
            "priority_key": list(self.priority_key),
            "priority_breakdown": priority_breakdown(self.priority_key),
            "prediction": self.prediction.as_dict(),
            "label_status": self.label_status,
            "verified": self.verified,
            "protected": self.protected,
            "remaining_days": self.remaining_days,
            "can_return_on": self.can_return_on.isoformat(),
            "outcome": self.outcome,
            "group": self.group,
            "device": self.device,
            "parts": list(self.parts),
            "handover_on": self.handover_on.isoformat() if self.handover_on else None,
            "explanation": self.explanation,
            "blockers": list(self.blockers),
            "commitment": self.commitment,
            "commitment_until": self.commitment_until.isoformat() if self.commitment_until else None,
        }


def _priority_key(
    request: Request,
    prediction: Prediction,
    label: Label,
    commitment: Reservation | None,
    protected: bool,
) -> tuple:
    """优先级元组，越靠前越优先。

    依次比较：旧承诺、规定的保护人群、失败证据、人员确认、后续使用时间、
    可信风险、可交接日期。``risk`` 只在有样本支撑时参与；人工测量的个案
    不进入自动分配，因此不会靠一个凭空的分数插队。
    """

    risk = prediction.risk if prediction.supported and prediction.risk is not None else -1.0
    return (
        1 if commitment else 0,
        1 if protected else 0,
        1 if label.status == "fit_failure" else 0,
        1 if request.verified else 0,
        request.remaining_days,
        risk,
        -request.can_return_on.toordinal(),
    )


PRIORITY_LABELS = (
    "旧承诺",
    "保护人群",
    "失败证据",
    "人员确认",
    "剩余使用日",
    "可信风险",
    "可交接日期",
)


def priority_breakdown(key: tuple) -> list[dict]:
    """把优先级元组翻译成可读的比较顺序。"""

    rows = []
    for index, (label, value) in enumerate(zip(PRIORITY_LABELS, key)):
        if index == 6:
            value = -value
        rows.append({"rank": index + 1, "field": label, "value": value})
    return rows


def _selection_key(key: tuple, request_id: str) -> tuple:
    """排序用：越大越优先；同分按编号稳定排序。"""

    return (
        -key[0],
        -key[1],
        -key[2],
        -key[3],
        -key[4],
        -key[5],
        -key[6],
        request_id,
    )


def _allocate(
    bundle: Bundle,
    order: Sequence[Request],
    snapshot: StockSnapshot,
    asof: date,
    reserved_devices: dict[str, str],
    reserved_parts: dict[str, str],
) -> dict[str, dict]:
    """按给定顺序尝试分配；返回每个个案的尝试记录。"""

    committed_ids = set(reserved_devices.values()) | set(reserved_parts.values())
    results: dict[str, dict] = {}
    for request in order:
        loan = bundle.loan(request.loan)
        options, width_issue = _options_for(bundle, request, snapshot, asof)
        dates = _feasible_dates(request, asof)
        if not dates:
            results[request.request] = {
                "outcome": "no_feasible_date",
                "blockers": [f"全部交接日期都早于可交回日期 {request.can_return_on.isoformat()}"],
            }
            continue
        if not options:
            if width_issue:
                results[request.request] = {
                    "outcome": "width_unverified",
                    "blockers": ["借出时未记录场地净宽，无法认可通过宽度"],
                }
            else:
                results[request.request] = {
                    "outcome": "no_compatible",
                    "blockers": ["没有同时满足站点、用途、尺寸和配件系列的组合"],
                }
            continue

        blockers: list[tuple[str, str]] = []
        allocated = False
        for day in dates:
            for option in options:
                device_id = option.device.device
                if device_id not in snapshot.available_devices:
                    blockers.append(("unavailable", f"{device_id} 已被领取且未再入库"))
                    continue
                if device_id in reserved_devices and reserved_devices[device_id] != request.request:
                    owner = reserved_devices[device_id]
                    kind = "commitment" if owner in committed_ids else "higher"
                    label = "旧承诺" if kind == "commitment" else "更高优先级"
                    blockers.append(
                        (kind, f"{device_id} 已保留给{label}安排 {owner}")
                    )
                    continue
                hold = _asset_hold_until(bundle, device_id, asof)
                if hold and hold[1] >= day:
                    blockers.append(
                        ("held", f"{device_id} 由保留 {hold[0].reservation} 占用至 {hold[1].isoformat()}")
                    )
                    continue
                chosen_parts = []
                part_problem: tuple[str, str] | None = None
                for category, candidates in sorted(option.parts_by_category.items()):
                    picked = None
                    for part_id in candidates:
                        if part_id in reserved_parts and reserved_parts[part_id] != request.request:
                            owner = reserved_parts[part_id]
                            kind = "commitment" if owner in committed_ids else "higher"
                            label = "旧承诺" if kind == "commitment" else "更高优先级"
                            part_problem = (
                                kind,
                                f"配件 {part_id} 已保留给{label}安排 {owner}",
                            )
                            continue
                        part_hold = _asset_hold_until(bundle, part_id, asof)
                        if part_hold and part_hold[1] >= day:
                            part_problem = (
                                "held",
                                f"配件 {part_id} 由保留 {part_hold[0].reservation} "
                                f"占用至 {part_hold[1].isoformat()}",
                            )
                            continue
                        picked = part_id
                        break
                    if picked is None:
                        part_problem = part_problem or (
                            "unavailable",
                            f"没有可用于类别 {category} 的已清点配件",
                        )
                        break
                    chosen_parts.append(picked)
                if part_problem and len(chosen_parts) < len(option.parts_by_category):
                    blockers.append((part_problem[0], f"{device_id}：{part_problem[1]}"))
                    continue
                if len(chosen_parts) != len(option.parts_by_category):
                    continue
                reserved_devices[device_id] = request.request
                for part_id in chosen_parts:
                    reserved_parts[part_id] = request.request
                results[request.request] = {
                    "outcome": "matched",
                    "device": device_id,
                    "parts": tuple(chosen_parts),
                    "handover_on": day,
                    "blockers": (),
                }
                allocated = True
                break
            if allocated:
                break
        if not allocated:
            kinds = {kind for kind, _message in blockers}
            if "commitment" in kinds:
                outcome = "existing_commitment_hold"
            elif "higher" in kinds:
                outcome = "higher_priority"
            elif "held" in kinds:
                outcome = "handover_conflict"
            else:
                # 兼容组合存在，但主件或配件尚未退回清点/已不可入库，属于当下匹配不到。
                outcome = "no_compatible"
            results[request.request] = {
                "outcome": outcome,
                "blockers": tuple(dict.fromkeys(message for _kind, message in blockers)),
            }
    return results


def plan_exchanges(
    bundle: Bundle,
    predictions: dict[str, Prediction],
    labels: dict[str, Label],
    asof: date,
    *,
    model_name: str = "backoff",
) -> dict[str, Recommendation]:
    snapshot = stock_snapshot(bundle, asof)
    commitments = {
        request_id: _commitment_for(bundle, request_id, asof) for request_id in bundle.requests
    }
    keys = {
        request_id: _priority_key(
            request,
            predictions[request_id],
            labels[request.loan],
            commitments[request_id],
            bundle.loan(request.loan).protected,
        )
        for request_id, request in bundle.requests.items()
    }
    order = sorted(bundle.requests.values(), key=lambda item: _selection_key(keys[item.request], item.request))
    positions = {request.request: index + 1 for index, request in enumerate(order)}

    reserved_devices: dict[str, str] = {}
    reserved_parts: dict[str, str] = {}
    committed_ids = set()
    for request_id, commitment in commitments.items():
        if commitment is None:
            continue
        committed_ids.add(request_id)
        reserved_devices[commitment.device] = request_id
        for part_id in commitment.parts:
            reserved_parts[part_id] = request_id

    # 人工测量的配置不进入自动分配，保留库存给有样本支撑的个案。
    auto_order = [
        request
        for request in order
        if request.request not in committed_ids
        and not predictions[request.request].manual
    ]
    attempts = _allocate(
        bundle,
        auto_order,
        snapshot,
        asof,
        reserved_devices,
        reserved_parts,
    )

    recommendations: dict[str, Recommendation] = {}
    for request in order:
        request_id = request.request
        prediction = predictions[request_id]
        label = labels[request.loan]
        commitment = commitments[request_id]
        key = keys[request_id]
        if commitment is not None:
            outcome = "existing_commitment"
            explanation = _commitment_explanation(bundle, request, commitment, prediction)
            device = commitment.device
            parts = commitment.parts
            handover = commitment.until_on
            blockers: tuple[str, ...] = ()
        elif prediction.manual:
            outcome = "manual_measurement"
            explanation = (
                "现有样本不足以支撑该配置的风险判断，先安排人工测量；"
                "在测量完成前不给出有把握的风险值。"
            )
            device = None
            parts = ()
            handover = None
            blockers = (prediction.note,)
        else:
            attempt = attempts.get(request_id, {"outcome": "no_compatible", "blockers": ()})
            outcome = attempt["outcome"]
            device = attempt.get("device")
            parts = attempt.get("parts", ())
            handover = attempt.get("handover_on")
            blockers = tuple(attempt.get("blockers", ()))
            explanation = _attempt_explanation(outcome, device, parts, handover, blockers)
        recommendations[request_id] = Recommendation(
            request=request_id,
            loan=request.loan,
            site=bundle.loan(request.loan).site,
            priority_position=positions[request_id],
            priority_key=key,
            prediction=prediction,
            label_status=label.status,
            verified=request.verified,
            protected=bundle.loan(request.loan).protected,
            remaining_days=request.remaining_days,
            can_return_on=request.can_return_on,
            outcome=outcome,
            group=GROUPS.get(outcome, outcome),
            device=device,
            parts=tuple(parts),
            handover_on=handover,
            explanation=explanation,
            blockers=blockers,
            commitment=commitment.reservation if commitment else None,
            commitment_until=commitment.until_on if commitment else None,
        )
    return recommendations


def _commitment_explanation(
    bundle: Bundle, request: Request, commitment: Reservation, prediction: Prediction
) -> str:
    loan = bundle.loan(request.loan)
    device = bundle.device(commitment.device)
    compatible = (
        device.site == loan.site
        and loan.use in device.uses
        and device.covers(loan.reach)
        and (loan.doorway is None or loan.doorway >= device.width)
    )
    parts_ok = all(
        bundle.part(part_id).category in loan.needs
        and device.family in bundle.part(part_id).families
        for part_id in commitment.parts
    )
    if compatible and parts_ok:
        fit_note = "原组合仍在资料认可的兼容范围内"
    else:
        fit_note = "原组合不完全符合当前兼容范围，但承诺尚未过期"
    risk_note = (
        "当前预测认为该配置风险偏高，但已答应的领取组合在承诺过期或明确改签前继续保留。"
        if prediction.supported and (prediction.risk or 0) >= 0.5
        else "承诺尚未到期，继续保留原主件和共同配件。"
    )
    return (
        f"保留 {commitment.reservation}（{commitment.device} + "
        f"{'、'.join(commitment.parts) or '无配件'}）至 {commitment.until_on.isoformat()}；"
        f"{fit_note}。{risk_note}"
    )


def _attempt_explanation(
    outcome: str, device: str | None, parts: Sequence[str], handover: date | None, blockers: Sequence[str]
) -> str:
    if outcome == "matched":
        return (
            f"可在 {handover.isoformat()} 交接 {device} + "
            f"{'、'.join(parts) or '无配件'}；主件与全部所需配件当日均可兑现。"
        )
    if outcome == "no_compatible":
        return (
            "没有可兑现的兼容组合：可能没有同时满足站点、用途、可调尺寸、通过净宽和"
            "配件系列的组合，或唯一合适的组合已被领取/尚未退回清点，需要人工评估。"
        )
    if outcome == "width_unverified":
        return "存在尺寸合适的器具，但借出时没有场地净宽，无法认可实际调换，需要人工测量。"
    if outcome == "no_feasible_date":
        return "可交回旧件的日期晚于全部可交接日期，交接时间无法对上。"
    if outcome == "higher_priority":
        return "有兼容组合，但可用资产已保留给优先级更高的个案。"
    if outcome == "existing_commitment_hold":
        return "有兼容组合，但主件或配件正由尚未到期或改签的旧承诺占用。"
    return "有兼容组合，但在可交接日期不可兑现：主件或配件尚未退回清点、已不在库，或交接日对不上。"


def risk_only_plan(
    bundle: Bundle,
    predictions: dict[str, Prediction],
    labels: dict[str, Label],
    asof: date,
) -> dict[str, Recommendation]:
    """对照：只按可信风险从高到低排队，其余分配规则相同。"""

    snapshot = stock_snapshot(bundle, asof)
    commitments = {
        request_id: _commitment_for(bundle, request_id, asof) for request_id in bundle.requests
    }
    reserved_devices = {}
    reserved_parts = {}
    committed_ids = set()
    for request_id, commitment in commitments.items():
        if commitment is None:
            continue
        committed_ids.add(request_id)
        reserved_devices[commitment.device] = request_id
        for part_id in commitment.parts:
            reserved_parts[part_id] = request_id

    ordered = sorted(
        (
            request
            for request in bundle.requests.values()
            if request.request not in committed_ids
            and predictions[request.request].supported
        ),
        key=lambda item: (
            -(predictions[item.request].risk or 0.0),
            -predictions[item.request].support,
            item.request,
        ),
    )
    attempts = _allocate(
        bundle,
        ordered,
        snapshot,
        asof,
        reserved_devices,
        reserved_parts,
    )
    result = {}
    for index, request in enumerate(ordered, 1):
        attempt = attempts.get(request.request, {"outcome": "no_compatible", "blockers": ()})
        prediction = predictions[request.request]
        result[request.request] = Recommendation(
            request=request.request,
            loan=request.loan,
            site=bundle.loan(request.loan).site,
            priority_position=index,
            priority_key=(
                0,
                int(bundle.loan(request.loan).protected),
                int(predictions[request.request].risk and (predictions[request.request].risk or 0) >= 0.5),
                int(request.verified),
                request.remaining_days,
                prediction.risk or 0.0,
                -request.can_return_on.toordinal(),
            ),
            prediction=prediction,
            label_status=labels[request.loan].status,
            verified=request.verified,
            protected=bundle.loan(request.loan).protected,
            remaining_days=request.remaining_days,
            can_return_on=request.can_return_on,
            outcome=attempt["outcome"],
            group=GROUPS.get(attempt["outcome"], attempt["outcome"]),
            device=attempt.get("device"),
            parts=tuple(attempt.get("parts", ())),
            handover_on=attempt.get("handover_on"),
            explanation=_attempt_explanation(
                attempt["outcome"],
                attempt.get("device"),
                attempt.get("parts", ()),
                attempt.get("handover_on"),
                attempt.get("blockers", ()),
            ),
            blockers=tuple(attempt.get("blockers", ())),
            commitment=None,
            commitment_until=None,
        )
    return result


def compare_schedules(
    policy: dict[str, Recommendation],
    risk_only: dict[str, Recommendation],
    bundle: Bundle,
) -> list[dict]:
    rows = []
    for request_id in sorted(policy):
        left = policy[request_id]
        right = risk_only.get(request_id)
        if right is None:
            difference = "旧承诺或人工测量，不参与按风险排队"
        elif left.device != right.device or left.parts != right.parts or left.handover_on != right.handover_on:
            difference = "资源去向不同"
        elif left.outcome != right.outcome:
            difference = "结论不同"
        else:
            difference = "一致"
        rows.append(
            {
                "request": request_id,
                "site": left.site,
                "policy_outcome": left.outcome,
                "risk_only_outcome": right.outcome if right else None,
                "policy_assets": _assets(left),
                "risk_only_assets": _assets(right) if right else None,
                "difference": difference,
            }
        )
    return rows


def _assets(recommendation: Recommendation | None) -> str | None:
    if recommendation is None or recommendation.device is None:
        return None
    return recommendation.device + (
        f" + {'、'.join(recommendation.parts)}" if recommendation.parts else ""
    )
