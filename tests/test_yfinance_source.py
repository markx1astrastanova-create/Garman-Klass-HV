import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

from BACKEND import main


class YahooSourceTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2023-01-01", periods=700, tz="Asia/Jakarta", name="Date")
        prices = 100 + np.arange(700) * 0.1
        self.history = pd.DataFrame({
            "Open": prices,
            "High": prices + 2,
            "Low": prices - 1,
            "Close": prices + np.sin(np.arange(700)),
            "Volume": 1000,
        }, index=index)

    def test_every_existing_ticker_has_yahoo_symbol(self):
        self.assertEqual(len(main.get_tickers()), 20)
        self.assertEqual(main.TICKER_DB["JKSE"]["yahoo_symbol"], "^JKSE")
        self.assertEqual(main.TICKER_DB["HIS"]["yahoo_symbol"], "^HSI")
        self.assertEqual(main.TICKER_DB["VNINDEX"]["yahoo_symbol"], "^VNINDEX.VN")
        self.assertEqual(main.TICKER_DB["GC1!"]["yahoo_symbol"], "GC=F")
        self.assertTrue(all(info["yahoo_symbol"] for info in main.TICKER_DB.values()))

    @patch("BACKEND.main.yf.Ticker")
    def test_500_daily_bars_preserve_math_and_response(self, ticker):
        ticker.return_value.history.return_value = self.history
        rows = main.get_volatility("JKSE", window=60)
        ticker.assert_called_once_with("^JKSE")
        options = ticker.return_value.history.call_args.kwargs
        self.assertEqual(options["interval"], "1d")
        self.assertFalse(options["auto_adjust"])
        self.assertEqual(len(rows), 500)
        self.assertEqual(set(rows[0]), {"date", "Open", "High", "Low", "Close", "GK_Vol", "GK_Zscore"})
        self.assertEqual(rows[0]["date"], "2023-07-20")
        expected = main.calculate_zscore_metrics(self.history.tail(500), 60).fillna(0)
        self.assertAlmostEqual(rows[-1]["GK_Vol"], expected.iloc[-1]["GK_Vol"])
        self.assertAlmostEqual(rows[-1]["GK_Zscore"], expected.iloc[-1]["GK_Zscore"])
        json.dumps(rows, allow_nan=False)

    @patch("BACKEND.main.yf.Ticker")
    def test_empty_history_preserves_not_found(self, ticker):
        ticker.return_value.history.return_value = pd.DataFrame()
        with self.assertRaises(HTTPException) as caught:
            main.get_volatility("JKSE")
        self.assertEqual(caught.exception.status_code, 404)

    @patch("BACKEND.main.yf.Ticker")
    def test_missing_and_zero_prices_do_not_corrupt_math(self, ticker):
        history = self.history.copy()
        history.iloc[-1, history.columns.get_loc("Open")] = np.nan
        history.iloc[-2, history.columns.get_loc("Low")] = 0
        ticker.return_value.history.return_value = history
        rows = main.get_volatility("JKSE", n_bars=100)
        self.assertEqual(len(rows), 100)
        self.assertEqual(rows[-1]["date"], history.index[-3].strftime("%Y-%m-%d"))
        json.dumps(rows, allow_nan=False)

    @patch("BACKEND.main.yf.Ticker")
    def test_short_history_reports_source_limit_instead_of_zero_volatility(self, ticker):
        ticker.return_value.history.return_value = self.history.tail(1)
        with self.assertRaises(HTTPException) as caught:
            main.get_volatility("VNINDEX")
        self.assertEqual(caught.exception.status_code, 404)
        self.assertIn("1 bar OHLC valid", caught.exception.detail)

    @patch("BACKEND.main.yf.Ticker")
    def test_upstream_failure_remains_http_error(self, ticker):
        ticker.return_value.history.side_effect = RuntimeError("Yahoo unavailable")
        with self.assertRaises(HTTPException) as caught:
            main.get_volatility("JKSE")
        self.assertEqual(caught.exception.status_code, 500)


if __name__ == "__main__":
    unittest.main()
