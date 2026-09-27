"""Deterministic parsing and display of the first Vietnamese deadline grammar."""

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


try:
    VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
except ZoneInfoNotFoundError as error:  # pragma: no cover - host configuration
    raise RuntimeError("Missing IANA timezone data for Asia/Ho_Chi_Minh") from error


class DeadlineParseError(ValueError):
    """The supplied deadline text is outside the supported grammar."""


_ABSOLUTE_DATE_FIRST = re.compile(
    r"^(?P<day>[0-9]{1,2})/(?P<month>[0-9]{1,2})/(?P<year>[0-9]{4})\s+"
    r"(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})$",
    re.IGNORECASE,
)
_ABSOLUTE_TIME_FIRST = re.compile(
    r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})\s+ngày\s+"
    r"(?P<day>[0-9]{1,2})/(?P<month>[0-9]{1,2})/(?P<year>[0-9]{4})$",
    re.IGNORECASE,
)
_RELATIVE_COLON = re.compile(
    r"^(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})\s+"
    r"(?P<day_word>hôm nay|mai|ngày mai)$",
    re.IGNORECASE,
)
_RELATIVE_WORDS = re.compile(
    r"^(?P<hour>[0-9]{1,2})\s+giờ"
    r"(?:\s+(?P<minute>[0-9]{1,2})\s+phút)?"
    r"\s+(?P<period>sáng|chiều|tối)\s+"
    r"(?P<day_word>hôm nay|mai|ngày mai)$",
    re.IGNORECASE,
)


def vietnam_now(reference: datetime | None = None) -> datetime:
    """Return one timezone-aware reference instant in Vietnam time."""
    if reference is None:
        return datetime.now(VIETNAM_TIMEZONE)
    if not isinstance(reference, datetime) or reference.tzinfo is None:
        raise ValueError("reference must be a timezone-aware datetime")
    return reference.astimezone(VIETNAM_TIMEZONE)


def _relative_date(word: str, reference: datetime) -> date:
    return reference.date() + (
        timedelta(days=0) if word.casefold() == "hôm nay" else timedelta(days=1)
    )


def _clock(hour_text: str, minute_text: str | None, period: str | None) -> tuple[int, int]:
    hour = int(hour_text)
    minute = int(minute_text or 0)
    if minute > 59:
        raise DeadlineParseError("Phút phải nằm trong khoảng 00–59.")
    if period is None:
        if hour > 23:
            raise DeadlineParseError("Giờ phải nằm trong khoảng 0–23.")
        return hour, minute
    if not 1 <= hour <= 12:
        raise DeadlineParseError("Giờ kèm sáng/chiều/tối phải nằm trong khoảng 1–12.")
    normalized_period = period.casefold()
    if normalized_period == "sáng":
        return (0 if hour == 12 else hour), minute
    return (12 if hour == 12 else hour + 12), minute


def _timestamp(day: date, hour: int, minute: int) -> int:
    try:
        value = datetime(
            day.year,
            day.month,
            day.day,
            hour,
            minute,
            tzinfo=VIETNAM_TIMEZONE,
        )
    except ValueError as error:
        raise DeadlineParseError("Ngày hoặc giờ không hợp lệ.") from error
    return int(value.timestamp())


def parse_deadline(text: str, *, reference: datetime | None = None) -> int:
    """Parse the supported Vietnamese forms into Unix seconds."""
    if not isinstance(text, str) or not text.strip() or "\n" in text or "\r" in text:
        raise DeadlineParseError("Thời hạn phải là một dòng không trống.")
    value = re.sub(r"\s+", " ", text.strip())
    value = value.rstrip(".?!").rstrip()
    now = vietnam_now(reference)

    match = _ABSOLUTE_DATE_FIRST.fullmatch(value) or _ABSOLUTE_TIME_FIRST.fullmatch(value)
    if match is not None:
        try:
            day = date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError as error:
            raise DeadlineParseError("Ngày không hợp lệ.") from error
        hour, minute = _clock(match.group("hour"), match.group("minute"), None)
        return _timestamp(day, hour, minute)

    match = _RELATIVE_COLON.fullmatch(value)
    if match is not None:
        hour, minute = _clock(match.group("hour"), match.group("minute"), None)
        return _timestamp(_relative_date(match.group("day_word"), now), hour, minute)

    match = _RELATIVE_WORDS.fullmatch(value)
    if match is not None:
        hour, minute = _clock(
            match.group("hour"), match.group("minute"), match.group("period")
        )
        return _timestamp(_relative_date(match.group("day_word"), now), hour, minute)

    raise DeadlineParseError(
        "Không hiểu thời hạn. Hãy dùng ngày/giờ rõ ràng, ví dụ "
        "27/09/2026 08:00 hoặc 8 giờ sáng mai."
    )


def format_deadline(due_at: int) -> str:
    """Render Unix seconds in the fixed Vietnam display format."""
    if isinstance(due_at, bool) or not isinstance(due_at, int):
        raise ValueError("due_at must be an integer Unix timestamp")
    try:
        value = datetime.fromtimestamp(due_at, VIETNAM_TIMEZONE)
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError("due_at is outside the supported datetime range") from error
    return value.strftime("%d/%m/%Y %H:%M")
