"""读取并校验辅具借用资料。

资料按来源表以 JSON Lines 保存。这里只负责把资料解析成带日期类型的对象并检查
引用关系；业务判断（标签、预测、换配）放在其他模块。所有读取都是只读的，站点
更新资料后重新运行即可。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
from typing import Iterable


class DataError(ValueError):
    """资料格式或引用关系不符合约定。"""


def _date(value, where: str) -> date:
    if not isinstance(value, str):
        raise DataError(f"{where} 应为 YYYY-MM-DD 日期")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise DataError(f"{where} 不是有效日期：{value!r}") from error


def _text(value, where: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise DataError(f"{where} 应为非空文本")
    return value


def _int(value, where: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataError(f"{where} 应为整数")
    return value


def _bool(value, where: str) -> bool:
    if not isinstance(value, bool):
        raise DataError(f"{where} 应为 true/false")
    return value


def _text_list(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DataError(f"{where} 应为非空文本数组")
    return tuple(value)


@dataclass(frozen=True)
class Loan:
    loan: str
    person: str
    site: str
    borrowed_on: date
    borrow_end: date
    due: date
    use: str
    reach: int
    doorway: int | None
    needs: tuple[str, ...]
    previous: str | None
    protected: bool

    @property
    def planned_days(self) -> int:
        return (self.borrow_end - self.borrowed_on).days


@dataclass(frozen=True)
class Outcome:
    reason: str
    loan: str
    observed_on: date
    kind: str
    evidence: str


@dataclass(frozen=True)
class Review:
    revision: str
    reason: str
    observed_on: date
    replacement: str
    corrected_reach: int | None


@dataclass(frozen=True)
class Device:
    device: str
    family: str
    site: str
    uses: tuple[str, ...]
    size_low: int
    size_high: int
    width: int

    def covers(self, reach: int) -> bool:
        return self.size_low <= reach <= self.size_high


@dataclass(frozen=True)
class Part:
    part: str
    category: str
    families: tuple[str, ...]
    checked: bool


@dataclass(frozen=True)
class Request:
    request: str
    loan: str
    old_device: str
    old_parts: tuple[str, ...]
    verified: bool
    remaining_days: int
    exchange_dates: tuple[date, ...]
    can_return_on: date


@dataclass(frozen=True)
class Reservation:
    reservation: str
    request: str
    device: str
    parts: tuple[str, ...]
    until_on: date
    state: str


@dataclass(frozen=True)
class StockEvent:
    event: str
    observed_on: date
    asset: str
    kind: str
    request: str | None


@dataclass
class Bundle:
    """一次读取得到的全部资料。"""

    directory: Path
    manifest: dict[str, list[str]]
    loans: dict[str, Loan]
    outcomes: tuple[Outcome, ...]
    reviews: tuple[Review, ...]
    devices: dict[str, Device]
    parts: dict[str, Part]
    requests: dict[str, Request]
    reservations: tuple[Reservation, ...]
    stock_events: tuple[StockEvent, ...]
    warnings: list[str] = field(default_factory=list)

    def loan(self, loan_id: str) -> Loan:
        try:
            return self.loans[loan_id]
        except KeyError as error:
            raise DataError(f"找不到借用记录 {loan_id!r}") from error

    def request(self, request_id: str) -> Request:
        try:
            return self.requests[request_id]
        except KeyError as error:
            raise DataError(f"找不到待换配个案 {request_id!r}") from error

    def device(self, device_id: str) -> Device:
        try:
            return self.devices[device_id]
        except KeyError as error:
            raise DataError(f"找不到器具 {device_id!r}") from error

    def part(self, part_id: str) -> Part:
        try:
            return self.parts[part_id]
        except KeyError as error:
            raise DataError(f"找不到配件 {part_id!r}") from error


def _read_jsonl(path: Path, columns: list[str]) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"{path.name} 第 {number} 行不是有效 JSON：{error.msg}") from error
            if not isinstance(record, dict) or record.keys() != set(columns):
                raise DataError(f"{path.name} 第 {number} 行的字段与登记格式不符")
            records.append(record)
    return records


def _unique(records: Iterable, key: str, table: str) -> dict:
    result = {}
    for record in records:
        identifier = getattr(record, key)
        if identifier in result:
            raise DataError(f"{table} 中编号 {identifier!r} 重复")
        result[identifier] = record
    return result


def read_bundle(directory: str | Path = "data") -> Bundle:
    """读取一个资料目录。缺表或字段不符时抛出 DataError。"""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DataError(f"无法读取 {manifest_path}：{error}") from error
    except json.JSONDecodeError as error:
        raise DataError(f"{manifest_path.name} 不是有效 JSON：{error.msg}") from error
    if not isinstance(manifest, dict):
        raise DataError("manifest.json 应为表名到列名的对象")

    raw = {}
    for name, columns in manifest.items():
        if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
            raise DataError(f"manifest.json 中 {name!r} 的列名应为文本数组")
        path = directory / f"{name}.jsonl"
        if not path.exists():
            raise DataError(f"缺少资料表 {path.name}")
        raw[name] = _read_jsonl(path, columns)

    warnings: list[str] = []
    required = {"loans", "outcomes", "reviews", "devices", "parts", "requests", "reservations", "stock_events"}
    missing = required - set(raw)
    if missing:
        raise DataError(f"manifest.json 缺少资料表：{', '.join(sorted(missing))}")

    loans = {}
    for row in raw["loans"]:
        lid = _text(row["loan"], "loans.loan")
        if lid in loans:
            raise DataError(f"loans 中编号 {lid!r} 重复")
        loans[lid] = Loan(
            loan=lid,
            person=_text(row["person"], f"loans[{lid}].person"),
            site=_text(row["site"], f"loans[{lid}].site"),
            borrowed_on=_date(row["borrowed_on"], f"loans[{lid}].borrowed_on"),
            borrow_end=_date(row["borrow_end"], f"loans[{lid}].borrow_end"),
            due=_date(row["due"], f"loans[{lid}].due"),
            use=_text(row["use"], f"loans[{lid}].use"),
            reach=_int(row["reach"], f"loans[{lid}].reach"),
            doorway=_int(row["doorway"], f"loans[{lid}].doorway", optional=True),
            needs=_text_list(row["needs"], f"loans[{lid}].needs"),
            previous=_text(row["previous"], f"loans[{lid}].previous", optional=True),
            protected=_bool(row["protected"], f"loans[{lid}].protected"),
        )

    outcomes = []
    for row in raw["outcomes"]:
        outcomes.append(
            Outcome(
                reason=_text(row["reason"], "outcomes.reason"),
                loan=_text(row["loan"], "outcomes.loan"),
                observed_on=_date(row["observed_on"], "outcomes.observed_on"),
                kind=_text(row["kind"], "outcomes.kind"),
                evidence=_text(row["evidence"], "outcomes.evidence"),
            )
        )
    outcome_by_reason = _unique(outcomes, "reason", "outcomes")

    reviews = []
    for row in raw["reviews"]:
        reviews.append(
            Review(
                revision=_text(row["revision"], "reviews.revision"),
                reason=_text(row["reason"], "reviews.reason"),
                observed_on=_date(row["observed_on"], "reviews.observed_on"),
                replacement=_text(row["replacement"], "reviews.replacement"),
                corrected_reach=_int(
                    row["corrected_reach"], "reviews.corrected_reach", optional=True
                ),
            )
        )
    _unique(reviews, "revision", "reviews")

    devices = {}
    for row in raw["devices"]:
        did = _text(row["device"], "devices.device")
        if did in devices:
            raise DataError(f"devices 中编号 {did!r} 重复")
        low = _int(row["size_low"], f"devices[{did}].size_low")
        high = _int(row["size_high"], f"devices[{did}].size_high")
        width = _int(row["width"], f"devices[{did}].width")
        if low > high:
            raise DataError(f"devices[{did}] 的 size_low 大于 size_high")
        if width <= 0:
            raise DataError(f"devices[{did}] 的 width 应为正数")
        devices[did] = Device(
            device=did,
            family=_text(row["family"], f"devices[{did}].family"),
            site=_text(row["site"], f"devices[{did}].site"),
            uses=_text_list(row["uses"], f"devices[{did}].uses"),
            size_low=low,
            size_high=high,
            width=width,
        )

    parts = {}
    for row in raw["parts"]:
        pid = _text(row["part"], "parts.part")
        if pid in parts:
            raise DataError(f"parts 中编号 {pid!r} 重复")
        parts[pid] = Part(
            part=pid,
            category=_text(row["category"], f"parts[{pid}].category"),
            families=_text_list(row["families"], f"parts[{pid}].families"),
            checked=_bool(row["checked"], f"parts[{pid}].checked"),
        )

    known_uses = {use for device in devices.values() for use in device.uses}
    for loan in loans.values():
        if loan.use not in known_uses:
            warnings.append(f"借用 {loan.loan} 的用途 {loan.use!r} 不在任何器具的合规用途内")
        if loan.borrowed_on > loan.borrow_end:
            raise DataError(f"loans[{loan.loan}] 的借出日期晚于正常归还日期")
        if loan.borrow_end > loan.due:
            warnings.append(f"借用 {loan.loan} 的正常归还日期晚于复核期限")
    for loan in loans.values():
        if loan.previous is not None:
            if loan.previous not in loans:
                raise DataError(f"loans[{loan.loan}].previous 指向不存在的借用 {loan.previous!r}")
    for outcome in outcomes:
        if outcome.loan not in loans:
            raise DataError(f"outcomes[{outcome.reason}] 指向不存在的借用 {outcome.loan!r}")
        if outcome.kind not in {"fit_confirmed", "fit_failure", "planned_short"}:
            warnings.append(f"结果 {outcome.reason} 的类型 {outcome.kind!r} 未在约定内")
    for review in reviews:
        if review.reason not in outcome_by_reason:
            raise DataError(f"reviews[{review.revision}] 指向不存在的复核理由 {review.reason!r}")

    requests = {}
    for row in raw["requests"]:
        qid = _text(row["request"], "requests.request")
        if qid in requests:
            raise DataError(f"requests 中编号 {qid!r} 重复")
        old_parts = _text_list(row["old_parts"], f"requests[{qid}].old_parts")
        exchange_dates = tuple(
            _date(value, f"requests[{qid}].exchange_dates") for value in row["exchange_dates"]
        )
        requests[qid] = Request(
            request=qid,
            loan=_text(row["loan"], f"requests[{qid}].loan"),
            old_device=_text(row["old_device"], f"requests[{qid}].old_device"),
            old_parts=old_parts,
            verified=_bool(row["verified"], f"requests[{qid}].verified"),
            remaining_days=_int(row["remaining_days"], f"requests[{qid}].remaining_days"),
            exchange_dates=exchange_dates,
            can_return_on=_date(row["can_return_on"], f"requests[{qid}].can_return_on"),
        )
    for request in requests.values():
        if request.loan not in loans:
            raise DataError(f"requests[{request.request}] 指向不存在的借用 {request.loan!r}")
        for part in request.old_parts:
            if part not in parts:
                raise DataError(f"requests[{request.request}].old_parts 含不存在的配件 {part!r}")

    reservations = []
    for row in raw["reservations"]:
        reservations.append(
            Reservation(
                reservation=_text(row["reservation"], "reservations.reservation"),
                request=_text(row["request"], "reservations.request"),
                device=_text(row["device"], "reservations.device"),
                parts=_text_list(row["parts"], "reservations.parts"),
                until_on=_date(row["until_on"], "reservations.until_on"),
                state=_text(row["state"], "reservations.state"),
            )
        )
    _unique(reservations, "reservation", "reservations")
    for reservation in reservations:
        if reservation.request not in requests:
            raise DataError(f"reservations[{reservation.reservation}] 指向不存在的个案 {reservation.request!r}")
        if reservation.device not in devices:
            raise DataError(f"reservations[{reservation.reservation}] 指向不存在的器具 {reservation.device!r}")
        for part in reservation.parts:
            if part not in parts:
                raise DataError(f"reservations[{reservation.reservation}] 指向不存在的配件 {part!r}")

    stock_events = []
    for row in raw["stock_events"]:
        stock_events.append(
            StockEvent(
                event=_text(row["event"], "stock_events.event"),
                observed_on=_date(row["observed_on"], "stock_events.observed_on"),
                asset=_text(row["asset"], "stock_events.asset"),
                kind=_text(row["kind"], "stock_events.kind"),
                request=_text(row["request"], "stock_events.request", optional=True),
            )
        )
    _unique(stock_events, "event", "stock_events")
    for event in stock_events:
        if event.asset not in devices and event.asset not in parts:
            raise DataError(f"stock_events[{event.event}] 的资产 {event.asset!r} 不在器具或配件表中")
        if event.request is not None and event.request not in requests:
            raise DataError(f"stock_events[{event.event}] 指向不存在的个案 {event.request!r}")
        if event.kind not in {"returned", "checked", "collected"}:
            warnings.append(f"库存事件 {event.event} 的类型 {event.kind!r} 未在约定内")

    return Bundle(
        directory=directory,
        manifest=manifest,
        loans=loans,
        outcomes=tuple(outcomes),
        reviews=tuple(reviews),
        devices=devices,
        parts=parts,
        requests=requests,
        reservations=tuple(reservations),
        stock_events=tuple(stock_events),
        warnings=warnings,
    )
