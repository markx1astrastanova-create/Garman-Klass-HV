import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from streamlit_backend import dispatch


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.backend = SimpleNamespace(
            get_tickers=lambda: [{"ticker": "JKSE"}],
            get_volatility=lambda **kwargs: self.calls.append(kwargs) or [{"Close": 100}],
        )

    def request(self, path):
        return dispatch({"id": "test", "path": path}, self.backend)

    def test_tickers(self):
        self.assertEqual(self.request("/api/tickers")["body"], [{"ticker": "JKSE"}])

    def test_volatility(self):
        self.assertEqual(self.request("/api/volatility?ticker=BBCA&window=20")["status"], 200)
        self.assertEqual(self.calls, [{"ticker": "BBCA", "window": 20}])

    def test_invalid_requests_do_not_call_backend(self):
        for path in ("/api/volatility?window=0", "/api/volatility?window=no", "/other", "https://example.com/api/tickers"):
            self.assertGreaterEqual(self.request(path)["status"], 400)
        self.assertEqual(self.calls, [])

    def test_backend_errors_reach_frontend(self):
        def fail(**kwargs):
            raise HTTPException(status_code=404, detail="Ticker not found in registry")
        self.backend.get_volatility = fail
        reply = self.request("/api/volatility?ticker=UNKNOWN")
        self.assertEqual(reply["status"], 404)
        self.assertEqual(reply["body"]["detail"], "Ticker not found in registry")


if __name__ == "__main__":
    unittest.main()
