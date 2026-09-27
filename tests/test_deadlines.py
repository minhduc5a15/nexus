import unittest
from datetime import datetime

from nexus.core.deadlines import (
    DeadlineParseError,
    VIETNAM_TIMEZONE,
    format_deadline,
    parse_deadline,
)


class DeadlineParserTests(unittest.TestCase):
    def setUp(self):
        self.reference = datetime(2026, 9, 26, 23, 30, tzinfo=VIETNAM_TIMEZONE)

    def assertDeadline(self, text, expected):
        timestamp = parse_deadline(text, reference=self.reference)
        self.assertEqual(format_deadline(timestamp), expected)

    def test_absolute_supported_forms_and_leap_year(self):
        self.assertDeadline("27/09/2026 08:00", "27/09/2026 08:00")
        self.assertDeadline("08:00 ngày 27/09/2026", "27/09/2026 08:00")
        self.assertDeadline("29/02/2028 23:59", "29/02/2028 23:59")

    def test_relative_colon_forms_use_the_injected_reference(self):
        self.assertDeadline("08:00 hôm nay", "26/09/2026 08:00")
        self.assertDeadline("08:00 mai", "27/09/2026 08:00")
        self.assertDeadline("08:00 ngày mai", "27/09/2026 08:00")

    def test_vietnamese_clock_converts_periods(self):
        self.assertDeadline("8 giờ sáng mai", "27/09/2026 08:00")
        self.assertDeadline("8 giờ 30 phút tối ngày mai", "27/09/2026 20:30")
        self.assertDeadline("12 giờ sáng mai", "27/09/2026 00:00")
        self.assertDeadline("12 giờ chiều mai", "27/09/2026 12:00")

    def test_past_deadlines_are_allowed(self):
        self.assertDeadline("01/01/2020 00:00", "01/01/2020 00:00")
        self.assertDeadline("08:00 hôm nay", "26/09/2026 08:00")

    def test_invalid_or_unsupported_forms_are_rejected(self):
        cases = (
            "", "   ", "mai", "08:00", "thứ hai tuần sau", "8h mai",
            "31/02/2026 08:00", "29/02/2027 08:00", "24:00 mai",
            "8 giờ 60 phút mai", "13 giờ tối mai", "20 giờ mai", "8 giờ mai", "08:00 UTC",
            "08:00 mai\n09:00 ngày mai", None,
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(DeadlineParseError):
                parse_deadline(value, reference=self.reference)

    def test_reference_must_be_timezone_aware(self):
        with self.assertRaises(ValueError):
            parse_deadline(
                "08:00 mai", reference=datetime(2026, 9, 26, 12, 0)
            )

    def test_format_rejects_non_integer_timestamp(self):
        for value in (True, 1.5, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                format_deadline(value)


if __name__ == "__main__":
    unittest.main()
