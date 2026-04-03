"""Tests for repository path helpers."""

import unittest

from gsebm.paths import get_data_dir, get_repo_root


class PathTest(unittest.TestCase):
    def test_get_repo_root_contains_pyproject(self) -> None:
        repo_root = get_repo_root()
        self.assertTrue((repo_root / "pyproject.toml").exists())

    def test_get_data_dir_points_inside_repo(self) -> None:
        data_dir = get_data_dir()
        self.assertEqual(data_dir.parent, get_repo_root())
        self.assertEqual(data_dir.name, "data")


if __name__ == "__main__":
    unittest.main()
