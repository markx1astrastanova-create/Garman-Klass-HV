from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st

# st.set_page_config WAJIB berada di paling atas sebelum perintah st lainnya
st.set_page_config(page_title="Garman Klass Volatility", layout="wide")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import os

# ==============================================================================
# KONFIGURASI
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IDX_CSV_PATH = os.path.join(BASE_DIR, "data", "idx_tickers.csv")


DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_WINDOW = 60

# ==============================================================================
# FUNGSI UTAMA
# ==============================================================================

@st.cache_data(ttl=86400, show_spinner="Memuat daftar saham IDX...")
def get_idx_stocks() -> pd.DataFrame:
    """Membaca CSV dengan toleransi delimiter; return DataFrame (Ticker, Nama, Type, Region)."""
    try:
        df = pd.read_csv(IDX_CSV_PATH, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        required = {'Ticker', 'Nama'}
        if not required.issubset(df.columns):
            st.error(f"Kolom wajib tidak ditemukan! Kolom: {df.columns.tolist()}")
            return pd.DataFrame()

        df = df.dropna(subset=['Ticker', 'Nama'])
        for c in ('Type', 'Region'):
            if c not in df.columns:
                df[c] = 'Other'
        return df
    except FileNotFoundError:
        st.warning(f"File {IDX_CSV_PATH} tidak ditemukan.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return pd.DataFrame()

def calculate_gk_volatility(df: pd.DataFrame) -> pd.Series:
    ln_open = np.log(df['Open'])
    c = np.log(df['Close']) - ln_open
    h = np.log(df['High']) - ln_open
    l = np.log(df['Low']) - ln_open
    gk_var = 0.511 * (h - l)**2 - 0.019 * (c * (h + l) - 2 * h * l) - 0.383 * (c**2)
    return np.sqrt(np.maximum(gk_var, 0))

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_data_batch(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    data = yf.download(tickers, start=start, end=end,
                       progress=False, auto_adjust=False, group_by='ticker')

    if len(tickers) == 1:
        if isinstance(data, pd.DataFrame) and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(-1)
            return {tickers[0]: data}
        return {tickers[0]: pd.DataFrame()}

    result = {}
    for t in tickers:
        df = data.get(t)
        if df is None or df.empty:
            result[t] = pd.DataFrame()
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        result[t] = df
    return result

def calculate_zscore_metrics(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df.copy()
    df['GK_Vol'] = calculate_gk_volatility(df)
    rolling = df['GK_Vol'].rolling(window=window)
    df['GK_Zscore_raw'] = (df['GK_Vol'] - rolling.mean()) / rolling.std()
    df['GK_Zscore'] = df['GK_Zscore_raw'].ewm(span=3, adjust=False).mean()
    return df

def plot_single_asset(ticker: str, nama: str, region: str, type_: str,
                      df: pd.DataFrame, window: int) -> None:
    """Satu grafik go.Figure dengan dual y-axis: Harga (atas) & Z-Score (bawah)."""
    if df.empty:
        return

    fig = go.Figure()

    # ── Trace Harga (panel atas) ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        name=f'{ticker} Close',
        line=dict(color='#4a90d9'),
        yaxis='y',
        hovertemplate='%{y:,.0f}<extra></extra>'
    ))

    # ── Trace Z-Score (panel bawah) ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df['GK_Zscore'],
        name=f'{ticker} Z-Score',
        line=dict(color='#e74c3c', width=1.5),
        yaxis='y2',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))

    # ── Threshold lines ──────────────────────────────────────────────────────
    fig.add_shape(type='line', xref='paper', x0=0, x1=1,
                  yref='y2', y0=2.5, y1=2.5,
                  line=dict(dash='dash', color='orange', width=1))
    fig.add_shape(type='line', xref='paper', x0=0, x1=1,
                  yref='y2', y0=0, y1=0,
                  line=dict(color='white', width=0.5), opacity=0.2)
    fig.add_shape(type='line', xref='paper', x0=0, x1=1,
                  yref='y2', y0=-1.0, y1=-1.0,
                  line=dict(dash='dash', color='green', width=1))

    # ── Annotations ─────────────────────────────────────────────────────────
    fig.add_annotation(xref='paper', x=0.01, yref='y2', y=2.5,
        text='VOL_EXPANSION (Z > 2.5)', showarrow=False,
        font=dict(color='orange', size=11), yanchor='bottom', xanchor='left')
    fig.add_annotation(xref='paper', x=0.01, yref='y2', y=-1.0,
        text='VOL_COMPRESSION (Z < -1.0)', showarrow=False,
        font=dict(color='green', size=11), yanchor='top', xanchor='left')

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f'{ticker} — {nama}',
            x=0.5, xanchor='center', y=0.97, yanchor='top',
            font=dict(size=20)
        ),
        height=850,
        margin=dict(t=80, b=40, l=30, r=30),
        template='plotly_dark',
        hovermode='x unified',
        hoverlabel=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
        spikedistance=-1,
        hoverdistance=-1,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=12)
        ),
        dragmode='pan',
        xaxis=dict(
            anchor='y2',
            title_text='',
            showspikes=True, spikemode='across',
            spikesnap='cursor', spikecolor='gray',
            spikethickness=0.5, spikedash='dot',
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1M', step='month', stepmode='backward'),
                    dict(count=3, label='3M', step='month', stepmode='backward'),
                    dict(count=6, label='6M', step='month', stepmode='backward'),
                    dict(count=1, label='1Y', step='year', stepmode='backward'),
                    dict(label='ALL', step='all')
                ],
                bgcolor='#2a2a2a', activecolor='#4a90d9',
                font=dict(color='white'), x=0.0, y=1.12
            )
        ),
        yaxis=dict(
            title_text='Closing Price',
            side='left',
            domain=[0.27, 1.0],
            anchor='x',
            fixedrange=False,
            showspikes=True, spikemode='across',
            spikesnap='cursor', spikecolor='gray',
            spikethickness=0.5, spikedash='dot'
        ),
        yaxis2=dict(
            title_text=f'GK Z-Score ({window}d)',
            side='right',
            domain=[0.0, 0.27],
            anchor='x',
            fixedrange=False,
            showspikes=True, spikemode='across',
            spikesnap='cursor', spikecolor='gray',
            spikethickness=0.5, spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True, 'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape']
    })


# ==============================================================================
# APLIKASI STREAMLIT
# ==============================================================================

if __name__ == "__main__":
    st.title("Analisis Volatilitas Garman-Klass")

    # Muat data
    idx_df = get_idx_stocks()
    if idx_df.empty:
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Pengaturan Analisis")

    # Filter Region & Type
    all_regions = sorted(idx_df['Region'].dropna().unique().tolist())
    all_types   = sorted(idx_df['Type'].dropna().unique().tolist())

    selected_regions = st.sidebar.multiselect(
        "Filter Region", options=all_regions, default=all_regions)
    selected_types = st.sidebar.multiselect(
        "Filter Type", options=all_types, default=all_types)

    filtered_df = idx_df[
        idx_df['Region'].isin(selected_regions) &
        idx_df['Type'].isin(selected_types)
    ]

    if filtered_df.empty:
        st.warning("Tidak ada aset yang sesuai filter.")
        st.stop()

    # HANYA 1 ASET — pakai selectbox, bukan multiselect
    ticker_options = (filtered_df['Ticker'] + ' - ' + filtered_df['Nama']).tolist()
    # Default ke yang mengandung ^JKSE (IHSG)
    default_idx = next((i for i, o in enumerate(ticker_options) if o.startswith('^JKSE')), 0)
    selected_display = st.sidebar.selectbox(
        "Pilih Aset", options=ticker_options, index=default_idx)

    # Tanggal & Window
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Mulai", datetime(2019, 1, 1).date())
    with col2:
        end_date = st.date_input("Akhir", datetime.now().date())

    window = st.sidebar.slider("Periode Rolling Z-Score", 10, 100, 60)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Menghitung Z-Score dari Garman-Klass Volatility "
        "untuk mengidentifikasi periode ekspansi/kompresi volatilitas."
    )

    # ── Area Utama ───────────────────────────────────────────────────────────
    # Ekstrak ticker dan metadata
    selected_ticker = selected_display.split(' - ')[0]
    meta = filtered_df[filtered_df['Ticker'] == selected_ticker].iloc[0]
    nama_aset   = meta['Nama']
    region_aset = meta.get('Region', 'Unknown')
    type_aset   = meta.get('Type', 'Unknown')

    # Info aset
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("Kode Saham", selected_ticker)
    col_info2.metric("Region", region_aset)
    col_info3.metric("Type", type_aset)

    # Download
    with st.spinner(f"Mengunduh data {selected_ticker}..."):
        raw = get_historical_data_batch(
            [selected_ticker],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    df_src = raw.get(selected_ticker)
    if df_src is None or df_src.empty:
        st.error(f"Data tidak ditemukan untuk {selected_ticker}.")
        st.stop()

    df_processed = calculate_zscore_metrics(df_src, window)

    # Plot SATU aset
    plot_single_asset(selected_ticker, nama_aset, region_aset, type_aset,
                      df_processed, window)

    # Tabel data
    with st.expander("Lihat data mentah"):
        display_df = df_processed[['Close', 'GK_Zscore']].copy()
        display_df.columns = ['Closing Price', 'GK Z-Score']
        st.dataframe(display_df, use_container_width=True)