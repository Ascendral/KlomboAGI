from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import build_brain_core


class BuildBrainCoreTests(unittest.TestCase):
    def test_build_native_reports_missing_maturin(self) -> None:
        with patch("scripts.build_brain_core.maturin_available", return_value=False):
            result = build_brain_core.build_native()
        self.assertFalse(result["ok"])
        self.assertIn("maturin", result["reason"])

    def test_main_returns_error_when_native_required_and_build_fails(self) -> None:
        with patch(
            "scripts.build_brain_core.build_native",
            return_value={"ok": False, "reason": "missing maturin"},
        ):
            rc = build_brain_core.main(["--require-native"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
