import pandas as pd
import os
from datetime import datetime


def scrape_idx_tickers():
    """
    Scrape daftar saham IHSG dari Wikipedia Bahasa Indonesia.
    URL: 'Daftar perusahaan tercatat di Bursa Efek Indonesia'
    """
    url = (
        "https://id.wikipedia.org/wiki/"
        "Daftar_perusahaan_yang_tercatat_di_Bursa_Efek_Indonesia"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print("=" * 60)
    print("UPDATE MASTER TICKER – ETL SAHAM IHSG")
    print("=" * 60)
    print(f"[{timestamp()}] Memulai proses scraping...")
    print(f"[{timestamp()}] Mengakses URL: {url}")
    print("-" * 60)

    # Baca semua tabel dari halaman Wikipedia
    dfs = pd.read_html(url, storage_options=headers)
    print(f"[{timestamp()}] Ditemukan {len(dfs)} tabel di halaman.")

    # Cari tabel yang memiliki kolom 'Kode saham' atau 'Kode Saham'
    target_df = None
    for i, df in enumerate(dfs):
        # Normalize column names: strip whitespace and lowercase
        cols = [str(c).strip().lower() for c in df.columns]
        if 'kode saham' in cols or 'kode' in cols:
            target_df = df
            print(f"[{timestamp()}] Tabel ditemukan di indeks ke-{i}.")
            print(f"[{timestamp()}] Kolom tabel: {list(df.columns)}")
            break

    if target_df is None:
        raise ValueError(
            "Tidak ditemukan tabel yang mengandung kolom 'Kode Saham'."
        )

    # Bersihkan nama kolom (strip whitespace)
    target_df.columns = [str(c).strip() for c in target_df.columns]

    # Cari nama kolom yang sesuai (case-insensitive)
    kode_col = None
    nama_col = None
    for col in target_df.columns:
        col_lower = col.strip().lower()
        if 'kode' in col_lower:
            kode_col = col
        if 'nama' in col_lower:
            nama_col = col

    if kode_col is None or nama_col is None:
        raise ValueError(
            f"Kolom 'Kode Saham' atau 'Nama Perusahaan' tidak ditemukan. "
            f"Kolom tersedia: {list(target_df.columns)}"
        )

    print(f"[{timestamp()}] Menggunakan kolom:")
    print(f"       Kode Saham      : '{kode_col}'")
    print(f"       Nama Perusahaan : '{nama_col}'")

    # Ambil hanya kolom yang diperlukan
    tickers = target_df[[kode_col, nama_col]].copy()
    tickers.columns = ['Kode Saham', 'Nama Perusahaan']

    # Bersihkan data: hapus baris yang kode sahamnya kosong / NaN
    tickers = tickers.dropna(subset=['Kode Saham'])
    tickers['Kode Saham'] = (
        tickers['Kode Saham']
        .astype(str)
        .str.strip()
        .str.replace(r'^BEI:\s*', '', regex=True)   # hapus prefix "BEI: "
    )

    # Hapus baris yang bukan kode saham valid (misal header/subheader)
    tickers = tickers[tickers['Kode Saham'].str.len() >= 2]

    print(f"[{timestamp()}] Total data setelah pembersihan: "
          f"{len(tickers)} saham.")

    return tickers


def format_ticker(kode):
    """Tambahkan suffix .JK untuk kompatibilitas Yahoo Finance."""
    kode = kode.strip().upper()
    if not kode.endswith('.JK'):
        return f"{kode}.JK"
    return kode


def save_to_csv(df):
    """Simpan DataFrame ke data/idx_tickers.csv."""
    os.makedirs('data', exist_ok=True)

    # Format kode saham
    df['Kode Saham'] = df['Kode Saham'].apply(format_ticker)

    output_path = os.path.join('data', 'idx_tickers.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')

    print("-" * 60)
    print(f"[{timestamp()}] File CSV berhasil dibuat!")
    print(f"[{timestamp()}] Lokasi        : {os.path.abspath(output_path)}")
    print(f"[{timestamp()}] Jumlah saham  : {len(df)}")
    print(f"[{timestamp()}] 5 contoh pertama:")
    for _, row in df.head(5).iterrows():
        print(f"       {row['Kode Saham']:12s} – {row['Nama Perusahaan']}")
    print("-" * 60)
    print("PROSES SELESAI.")
    print("=" * 60)

    return output_path


def timestamp():
    """Return timestamp string untuk log."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def main():
    try:
        tickers_df = scrape_idx_tickers()
        save_to_csv(tickers_df)
    except Exception as e:
        print(f"[{timestamp()}] ERROR: {e}")
        print("=" * 60)
        return 1
    return 0


if __name__ == '__main__':
    exit(main())