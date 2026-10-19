from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class StartupTests(unittest.TestCase):
    def test_component_bundle_and_app_startup(self):
        root = Path(__file__).resolve().parents[1]
        bundle = root / "FRONTEND" / "streamlit_dist"
        self.assertTrue((bundle / "index.html").is_file())
        self.assertTrue(list((bundle / "assets").glob("*.js")))
        app = AppTest.from_file(str(root / "streamlit_app.py")).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
