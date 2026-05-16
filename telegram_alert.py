import os
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

# ==========================================
# KONFIGURASI — Baca dari Environment Variable
# ==========================================
# AMAN: Token & Chat ID tidak di-hardcode.
# Set environment variable sebelum menjalankan:
#   export TELEGRAM_BOT_TOKEN="your_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"
# Atau via GitHub Secrets + YAML workflow.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Path ke file CSV lokal daftar saham IDX.
# Menggunakan path relatif terhadap lokasi script ini agar robust
# baik dijalankan dari folder mana pun (lokal maupun GitHub Actions).
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
IDX_CSV_PATH = os.path.join(_SCRIPT_DIR, "..", "data", "idx_tickers.csv")

# Jendela periode hitungan
WINDOW = 60
# ==========================================

def load_tickers_from_csv() -> list[str]:
    """Membaca daftar kode saham dari file CSV dengan toleransi delimiter dan spasi kolom."""
    try:
        # sep=None + engine='python' agar Pandas deteksi otomatis delimiter ; atau ,
        df = pd.read_csv(IDX_CSV_PATH, sep=None, engine='python')
        
        # Bersihkan spasi di nama kolom
        df.columns = df.columns.str.strip()
        
        # Hapus kolom kosong (Unnamed) akibat delimiter berlebih
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Cari kolom: coba 'Kode Saham' dulu, fallback ke 'Ticker'
        ticker_col = None
        for col in ['Kode Saham', 'Ticker']:
            if col in df.columns:
                ticker_col = col
                break
        
        if ticker_col is None:
            print(f" Kolom ticker tidak ditemukan! Kolom saat ini: {df.columns.tolist()}")
            return []
        
        df = df.dropna(subset=[ticker_col])
        tickers = df[ticker_col].str.strip().tolist()
        print(f" Memuat {len(tickers)} saham dari {IDX_CSV_PATH}")
        return tickers
    except FileNotFoundError:
        print(f" File {IDX_CSV_PATH} tidak ditemukan. Tidak ada saham yang dipindai.")
        return []
    except Exception as e:
        print(f" Gagal membaca daftar saham IDX: {e}")
        return []

def send_telegram_message(message: str):
    """Mengirim pesan ke Telegram menggunakan Bot API.
    Jika pesan melebihi 4000 karakter, akan dipecah menjadi beberapa bagian."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(" Telegram Token atau Chat ID belum diatur. Set environment variable:")
        print("  TELEGRAM_BOT_TOKEN  dan  TELEGRAM_CHAT_ID")
        print("Pesan yang seharusnya dikirim:\n", message)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Telegram API memiliki batas 4096 karakter per pesan.
    # Kita potong menjadi beberapa bagian jika melebihi 4000 karakter (cadangan).
    MAX_LENGTH = 4000
    
    if len(message) <= MAX_LENGTH:
        parts = [message]
    else:
        parts = []
        current = message
        while len(current) > MAX_LENGTH:
            # Potong di newline terdekat agar tidak merusak format
            split_at = current.rfind('\n\n', 0, MAX_LENGTH)
            if split_at == -1:
                split_at = current.rfind('\n', 0, MAX_LENGTH)
            if split_at == -1:
                split_at = MAX_LENGTH
            parts.append(current[:split_at])
            current = current[split_at:].strip()
        if current:
            parts.append(current)
    
    for i, part in enumerate(parts):
        # Kirim tanpa parse_mode HTML untuk menghindari error parsing
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": part
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print(f" Alert bagian {i+1}/{len(parts)} berhasil dikirim ke Telegram!")
        except requests.exceptions.HTTPError as e:
            print(f" Gagal mengirim bagian {i+1}/{len(parts)}: {e}")
            print(f" Response body: {response.text[:500]}")
        except Exception as e:
            print(f" Gagal mengirim bagian {i+1}/{len(parts)}: {e}")

def calculate_gk_volatility(df: pd.DataFrame) -> pd.Series:
    """Menghitung Garman-Klass Volatility."""
    ln_open = np.log(df['Open'])
    c = np.log(df['Close']) - ln_open
    h = np.log(df['High']) - ln_open
    l = np.log(df['Low']) - ln_open
    
    gk_var = 0.511 * (h - l)**2 - 0.019 * (c * (h + l) - 2 * h * l) - 0.383 * (c**2)
    return np.sqrt(np.maximum(gk_var, 0))

def calculate_zscore_metrics(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Kalkulasi Z-Score EWMA."""
    df = df.copy()
    df['GK_Vol'] = calculate_gk_volatility(df)
    
    rolling_vol = df['GK_Vol'].rolling(window=window)
    df['GK_Zscore_raw'] = (df['GK_Vol'] - rolling_vol.mean()) / rolling_vol.std()
    df['GK_Zscore'] = df['GK_Zscore_raw'].ewm(span=3, adjust=False).mean()
    
    return df

def run_alert_system():
    """Menjalankan scan ke semua saham secara batch dan mengirim alert jika memenuhi syarat."""
    print(f"Memulai pemindaian saham pada {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    # =====================================================
    # 1. Muat semua ticker dari CSV lokal (hapus hardcode)
    # =====================================================
    all_tickers = load_tickers_from_csv()
    if not all_tickers:
        print("Tidak ada daftar saham yang bisa dipindai. Program berhenti.")
        return
    
    # Ambil data dari 4 bulan terakhir agar aman untuk perhitungan rolling 60 hari
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # =====================================================
    # 2. Batch Download — tarik semua data dalam SATU panggilan
    # =====================================================
    print(f" Mengunduh data untuk {len(all_tickers)} saham secara batch...")
    data = yf.download(
        all_tickers,
        start=start_str,
        end=end_str,
        progress=False,
        auto_adjust=False,
        group_by='ticker'
    )
    print(" Unduhan selesai. Memproses hasil...")
    
    alerts = []
    processed_count = 0
    skipped_count = 0
    
    # =====================================================
    # 3. Loop melalui setiap ticker dari hasil batch
    # =====================================================
    for ticker in all_tickers:
        # Ambil DataFrame untuk ticker ini dari hasil batch
        df = data.get(ticker)
        
        # Skip jika data kosong
        if df is None or df.empty:
            skipped_count += 1
            continue
        
        # Flatten MultiIndex columns jika perlu
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        
        # Pastikan kolom OHLC yang diperlukan ada
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            skipped_count += 1
            continue
        
        # Skip jika data NaN terlalu banyak (kurang dari 30 baris valid)
        if df[required_cols].dropna().shape[0] < 30:
            skipped_count += 1
            continue
        
        # =============================================
        # 4. Kalkulasi Z-Score
        # =============================================
        df_processed = calculate_zscore_metrics(df, WINDOW)
        
        # Ambil data pada hari bursa terakhir
        latest_data = df_processed.iloc[-1]
        latest_zscore = latest_data['GK_Zscore']
        latest_date = df_processed.index[-1].strftime("%Y-%m-%d")
        
        # Skip jika Z-Score NaN
        if np.isnan(latest_zscore):
            skipped_count += 1
            continue
        
        processed_count += 1
        
        # =============================================
        # 5. Logika Alert
        # =============================================
        if latest_zscore >= 2.5:
            msg = (f"VOLATILITY EXPANSION\n"
                   f"Saham: {ticker}\n"
                   f"Z-Score: {latest_zscore:.2f} (>= 2.5)\n"
                   f"Tanggal: {latest_date}")
            alerts.append(msg)
            print(f"  EXPANSION: {ticker} (Z={latest_zscore:.2f})")
            
        elif latest_zscore <= -1.0:
            msg = (f"VOLATILITY COMPRESSION\n"
                   f"Saham: {ticker}\n"
                   f"Z-Score: {latest_zscore:.2f} (<= -1.0)\n"
                   f"Tanggal: {latest_date}")
            alerts.append(msg)
            print(f"  COMPRESSION: {ticker} (Z={latest_zscore:.2f})")
    
    print(f"\n Ringkasan: {processed_count} saham diproses, {skipped_count} dilewati, {len(alerts)} sinyal ditemukan.")
    
    # =====================================================
    # 6. Kirim SATU pesan batch ke Telegram
    # =====================================================
    if alerts:
        final_message = "Garman-Klass Volatility Alert\n\n" + "\n\n".join(alerts)
        send_telegram_message(final_message)
    else:
        # Jika tidak ada alert, tetap kirim laporan status aman ke Telegram
        safe_message = (
            "Garman-Klass Scanner Harian\n\n"
            f"{processed_count} saham dipindai, {skipped_count} dilewati.\n\n"
            "Semua saham yang dipantau saat ini berada dalam tingkat volatilitas normal "
            "(Z-Score di antara -1.0 dan 2.5).\n\n"
            "Tidak ada ekspansi atau kompresi signifikan hari ini."
        )
        send_telegram_message(safe_message)
        print("Laporan status aman telah dikirim ke Telegram.")

if __name__ == "__main__":
    run_alert_system()