"""Smoke tests for the package scaffold."""

import unittest

import gsebm


class PackageSmokeTest(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertTrue(gsebm.__version__)


if __name__ == "__main__":
    unittest.main()
