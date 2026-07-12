from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tvdatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np
import time
import random
import os

app = FastAPI(title="GKHV Quant Engine - TV Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi TvDatafeed secara anonim
try:
    tv = TvDatafeed()
except Exception as e:
    print(f"Gagal inisialisasi TradingView Datafeed: {e}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDX_CSV_PATH = os.path.join(BASE_DIR, "data", "idx_tickers.csv")

def get_idx_stocks() -> pd.DataFrame:
    try:
        df = pd.read_csv(IDX_CSV_PATH, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(subset=['Ticker', 'Nama'])
        return df
    except Exception as e:
        print(f"Gagal membaca CSV: {e}")
        return pd.DataFrame()

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
    df = get_idx_stocks()
    if df.empty:
        return []
    return df.to_dict(orient="records")

@app.get("/api/volatility")
def get_volatility(ticker: str, window: int = 60, n_bars: int = 500):
    try:
        # ⚠️ CRITICAL: Jeda waktu Anti-DDoS sebelum menembak TradingView
        time.sleep(random.uniform(1.5, 3.0))
        
        # Bersihkan format ticker (Contoh: BBRI.JK -> BBRI)
        clean_ticker = ticker.split('.')[0].upper()
        if clean_ticker in ["JKSE", "^JKSE", "COMPOSITE"]:
            clean_ticker = "COMPOSITE"
            
        # Ambil data dari TradingView
        df_tv = tv.get_hist(symbol=clean_ticker, exchange='IDX', interval=Interval.in_daily, n_bars=n_bars)
        
        if df_tv is None or df_tv.empty:
            raise HTTPException(status_code=404, detail=f"Data untuk {clean_ticker} tidak ditemukan di TradingView.")
            
        # Standarisasi kolom dari tvdatafeed ke Math Engine
        df_tv = df_tv.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
        })
        
        # Hitung Matrik Kuantitatif (Meilijson S2)
        df_processed = calculate_zscore_metrics(df_tv, window)
        
        # Reset indeks datetime untuk konversi JSON
        df_processed = df_processed.reset_index()
        # tvdatafeed mengembalikan index 'datetime'
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
