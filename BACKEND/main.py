from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
import yfinance as yf
import pandas as pd
import numpy as np
import os

app = FastAPI(title="GKHV Quant Engine - Yahoo Finance")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDX_CSV_PATH = os.path.join(BASE_DIR, "data", "idx_tickers.csv")

def load_ticker_db() -> dict:
    try:
        df = pd.read_csv(IDX_CSV_PATH, sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        db = {}
        for _, row in df.iterrows():
            ticker_input = str(row.get('ticker', '')).strip().upper()
            if ticker_input and ticker_input != 'NAN':
                db[ticker_input] = {
                    "symbol": str(row.get('symbol', '')).strip().upper(),
                    "exchange": str(row.get('exchange', '')).strip().upper(),
                    "yahoo_symbol": str(row.get('yahoo_symbol', '')).strip(),
                    "name": str(row.get('name', '')),
                    "type": str(row.get('type', '')),
                    "region": str(row.get('region', '')),
                    "active": str(row.get('active', '')).strip().lower() == 'true'
                }
        return db
    except Exception as e:
        print(f"Gagal load TICKER_DB: {e}")
        return {}

TICKER_DB = load_ticker_db()

def resolve_ticker(ticker_input: str) -> dict:
    clean_ticker = ticker_input.strip().upper()
    info = TICKER_DB.get(clean_ticker)
    if not info:
        raise HTTPException(status_code=404, detail="Ticker not found in registry")
    return info

def calculate_meilijson_volatility(df: pd.DataFrame) -> pd.Series:
    ln_open = np.log(df['Open'])
    c = np.log(df['Close']) - ln_open
    h = np.log(df['High']) - ln_open
    l = np.log(df['Low']) - ln_open
    
    C = np.abs(c)
    H = np.where(c > 0, h, -l)
    L = np.where(c > 0, -l, -h)
    
    s1_sq = C**2
    s2_sq = 2 * (H - C)**2
    s3_sq = 2 * L**2
    s4_sq = 2 * C * (H - C - L)
    
    a1 = 0.1604
    a2 = 0.2736
    a3 = 0.2736
    a4 = 0.3652
    
    var_est = a1 * s1_sq + a2 * s2_sq + a3 * s3_sq + a4 * s4_sq
    return np.sqrt(np.maximum(var_est, 0))

def calculate_zscore_metrics(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df.copy()
    df['GK_Vol'] = calculate_meilijson_volatility(df)
    rolling = df['GK_Vol'].rolling(window=window)
    df['GK_Zscore_raw'] = (df['GK_Vol'] - rolling.mean()) / rolling.std()
    df['GK_Zscore'] = df['GK_Zscore_raw'].ewm(span=3, adjust=False).mean()
    return df

@app.get("/api/tickers")
def get_tickers():
    active_tickers = []
    for ticker, info in TICKER_DB.items():
        if info['active']:
            active_tickers.append({
                "ticker": ticker,
                "name": info["name"],
                "type": info["type"],
                "region": info["region"],
                "exchange": info["exchange"]
            })
    return active_tickers

@app.get("/api/volatility")
def get_volatility(ticker: str, window: int = 60, n_bars: int = 500):
    try:
        # Resolve ticker mapping
        info = resolve_ticker(ticker)
        symbol = info["yahoo_symbol"]
        if not symbol or symbol.lower() == 'nan':
            raise HTTPException(status_code=404, detail="Ticker belum memiliki simbol Yahoo Finance")
        if n_bars < 1:
            raise HTTPException(status_code=422, detail="n_bars must be positive")

        # Beri ruang untuk akhir pekan dan hari libur sebelum mengambil n_bars terakhir.
        start = (datetime.now(timezone.utc) - timedelta(days=n_bars * 2 + 30)).date()
        df_prices = yf.Ticker(symbol).history(
            start=start.isoformat(), interval="1d", auto_adjust=False,
            actions=False, timeout=20, raise_errors=True,
        )

        if df_prices is None or df_prices.empty:
            raise HTTPException(status_code=404, detail=f"Data untuk {symbol} tidak ditemukan di Yahoo Finance.")

        # Yahoo sudah memakai nama kolom OHLC yang dibutuhkan Math Engine.
        ohlc = ['Open', 'High', 'Low', 'Close']
        df_prices = df_prices.dropna(subset=ohlc)
        df_prices = df_prices.loc[(df_prices[ohlc] > 0).all(axis=1)].sort_index().tail(n_bars)
        if df_prices.empty:
            raise HTTPException(status_code=404, detail=f"Data OHLC untuk {symbol} tidak tersedia di Yahoo Finance.")
        if len(df_prices) < window:
            raise HTTPException(status_code=404, detail=f"Yahoo Finance hanya menyediakan {len(df_prices)} bar OHLC valid untuk {symbol}; minimal {window} bar diperlukan.")
        
        # Hitung Matrik Kuantitatif (Meilijson S2)
        df_processed = calculate_zscore_metrics(df_prices, window)
        
        # Reset indeks datetime untuk konversi JSON
        df_processed = df_processed.reset_index()
        # Terima nama index tanggal dari penyedia data.
        if 'datetime' in df_processed.columns:
            df_processed['date'] = df_processed['datetime'].dt.strftime('%Y-%m-%d')
        else:
            df_processed['date'] = df_processed.iloc[:, 0].dt.strftime('%Y-%m-%d')
            
        # Bersihkan kolom dari nilai NaN akibat rolling window agar JSON valid
        df_processed = df_processed.fillna(0)
        
        # Output format sesuai spesifikasi: JSON Array of Objects
        data_json = df_processed[['date', 'Open', 'High', 'Low', 'Close', 'GK_Vol', 'GK_Zscore']].to_dict(orient="records")
        
        return data_json
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
