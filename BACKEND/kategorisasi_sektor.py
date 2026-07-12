import pandas as pd
import yfinance as yf
import os
from concurrent.futures import ThreadPoolExecutor
import time

# Path untuk file CSV yang sudah ada
INPUT_CSV = r"x:\Python Project\GKHV\data\idx_tickers.csv"
# Folder untuk menyimpan hasil
OUTPUT_DIR = r"x:\Python Project\RRG\data\sektor_ihsg"

# Membuat folder output jika belum ada
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_sector_info(ticker_symbol):
    """Fungsi untuk mengambil Sector dan Industry dari Yahoo Finance"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        sector = info.get('sector', None)
        industry = info.get('industry', None)
        return ticker_symbol, sector, industry
    except Exception as e:
        print(f"Error mengambil data {ticker_symbol}: {e}")
        return ticker_symbol, None, None

def map_to_idx_sector(yf_sector, yf_industry):
    """
    Fungsi untuk memetakan GICS Sector Yahoo Finance ke 11 Sektor Utama IHSG:
    1. Energy, 2. Basic Materials, 3. Industrials, 4. Konsumen primer, 
    5. Non primer, 6. Healthcare, 7. Finance, 8. Property, 
    9. Technology, 10. Infrastructure, 11. Transportation
    """
    if pd.isna(yf_sector) or yf_sector is None:
        return 'Unknown'
    
    yf_sector = str(yf_sector)
    yf_industry = str(yf_industry)
    
    if yf_sector == 'Energy': return '1. Energy'
    if yf_sector == 'Basic Materials': return '2. Basic Materials'
    if yf_sector == 'Consumer Defensive': return '4. Konsumen Primer'
    if yf_sector == 'Consumer Cyclical': return '5. Non Primer'
    if yf_sector == 'Healthcare': return '6. Healthcare'
    if yf_sector == 'Financial Services': return '7. Finance'
    if yf_sector == 'Real Estate': return '8. Property'
    if yf_sector == 'Technology': return '9. Technology'
    
    # Utilities dan Communication Services biasanya dimasukkan ke Infrastructure di IHSG
    if yf_sector in ['Utilities', 'Communication Services']: 
        return '10. Infrastructure'
        
    if yf_sector == 'Industrials':
        # Yahoo Finance menggabungkan Transportasi dan Infrastruktur ke dalam Industrials.
        # Kita pisahkan secara manual berdasarkan nama industri (Industry).
        transport_industries = [
            'Airlines', 'Marine Shipping', 'Railroads', 'Trucking', 
            'Integrated Freight & Logistics', 'Air Freight & Logistics',
            'Marine'
        ]
        if any(t in yf_industry for t in transport_industries):
            return '11. Transportation'
        
        infra_industries = [
            'Engineering & Construction', 'Infrastructure Operations'
        ]
        if any(i in yf_industry for i in infra_industries):
            return '10. Infrastructure'
            
        return '3. Industrials'
        
    return f'Lainnya ({yf_sector})'

def main():
    print("Membaca file CSV...")
    df = pd.read_csv(INPUT_CSV)
    
    # Filter hanya data yang Tipe-nya Stock dan di Region Indonesia (Saham IHSG)
    df_stocks = df[(df['Type'] == 'Stock') & (df['Region'] == 'Indonesia')].copy()
    tickers = df_stocks['Ticker'].tolist()
    
    print(f"Total saham IHSG yang akan diproses: {len(tickers)}")
    
    results = []
    # Menggunakan ThreadPoolExecutor untuk mempercepat proses ke Yahoo Finance
    print("Mulai mengunduh data dari Yahoo Finance (harap tunggu beberapa menit)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        for result in executor.map(get_sector_info, tickers):
            results.append(result)
            print(f"Berhasil: {result[0]} -> Sector: {result[1]}, Industry: {result[2]}")
            time.sleep(0.05) # Delay 50ms untuk menghindari di-blokir oleh Yahoo Finance
            
    # Membuat DataFrame dari hasil list
    sector_df = pd.DataFrame(results, columns=['Ticker', 'Sector Yahoo', 'Industry Yahoo'])
    
    # Menggabungkan data asli dengan data Sektor Yahoo
    df_final = pd.merge(df_stocks, sector_df, on='Ticker', how='left')
    
    # Mapping Sektor Yahoo menjadi Sektor IHSG
    df_final['Sektor IHSG'] = df_final.apply(lambda row: map_to_idx_sector(row['Sector Yahoo'], row['Industry Yahoo']), axis=1)
    
    # Menyimpan 1 file master utuh
    master_output = os.path.join(OUTPUT_DIR, "master_saham_dengan_sektor.csv")
    df_final.to_csv(master_output, index=False)
    print(f"\nBerhasil menyimpan 1 master data dengan sektor di: {master_output}")
    
    # Memisahkan ke masing-masing CSV per Sektor
    print("\nMemisahkan data ke masing-masing file per sektor...")
    for sektor, group in df_final.groupby('Sektor IHSG'):
        # Menghapus karakter yang dilarang pada penamaan file
        safe_name = str(sektor).replace('/', '_').replace(':', '').replace(' ', '_')
        output_file = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
        group.to_csv(output_file, index=False)
        print(f"- Tersimpan {len(group)} saham di file {output_file}")
        
    print("\nPROSES SELESAI! Semua file tersimpan di:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
