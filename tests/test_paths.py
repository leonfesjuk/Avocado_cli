import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from srs import paths


class XdgPathTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.env = {
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_STATE_HOME": str(root / "state"),
            "XDG_CACHE_HOME": str(root / "cache"),
        }
        self.patch = patch.dict(os.environ, self.env, clear=False)
        self.patch.start()
        os.environ.pop("AVOCADO_HOME", None)
        self.root = root

    def tearDown(self):
        self.patch.stop()
        self.tempdir.cleanup()

    def test_xdg_paths(self):
        self.assertEqual(
            paths.user_config_path(),
            self.root / "config" / "avocado" / "config.json",
        )
        self.assertEqual(
            paths.sessions_root(),
            self.root / "data" / "avocado" / "sessions",
        )
        self.assertEqual(paths.state_dir(), self.root / "state" / "avocado")
        self.assertEqual(paths.cache_dir(), self.root / "cache" / "avocado")

    def test_avocado_home_overrides_everything(self):
        home = self.root / "avocado-home"
        with patch.dict(os.environ, {"AVOCADO_HOME": str(home)}):
            self.assertEqual(paths.user_config_path(), home / "config.json")
            self.assertEqual(paths.sessions_root(), home / "sessions")
            self.assertEqual(paths.state_dir(), home / "state")
            self.assertEqual(paths.cache_dir(), home / "cache")

    def test_ensure_user_config_copies_template(self):
        template = self.root / "template.json"
        template.write_text('{"ok": true}\n', encoding="utf-8")

        target = paths.ensure_user_config(template)

        self.assertEqual(target, paths.user_config_path())
        self.assertTrue(target.exists())
        self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}\n')


if __name__ == "__main__":
    unittest.main()
