import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIGURASI DASHBOARD STREAMLIT
# ============================================================
st.set_page_config(
    page_title="PHARMASIX — Dashboard Inventaris Obat",
    page_icon="📦",
    layout="wide"
)

st.title("📦 PHARMASIX v3 — Dashboard Inventaris")
st.subheader("Sistem Prediksi Manajemen Stok Obat Berbasis Pasokan & AI (LSTM)")
st.markdown("---")

# ============================================================
# LOAD DATA ASLI DARI REPOSITORI
# ============================================================
@st.cache_data
def load_data():
    csv_file = "pharmasix_v3_laporan_stok_AI_DRIVEN.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # Menyesuaikan nama kolom jika ada perbedaan kapitalisasi/spasi
        df.columns = df.columns.str.strip().str.lower()
        return df
    else:
        st.error(f"File {csv_file} tidak ditemukan di repositori!")
        return pd.DataFrame()

df_stock = load_data()

if not df_stock.empty:
    # Mengidentifikasi nama kolom secara fleksibel agar tidak rentan error KeyError
    col_med = 'medicine' if 'medicine' in df_stock.columns else df_stock.columns[0]
    col_stock = 'stock_sekarang' if 'stock_sekarang' in df_stock.columns else (df_stock.columns[1] if len(df_stock.columns) > 1 else col_med)
    col_forecast = 'lstm_forecast_demand' if 'lstm_forecast_demand' in df_stock.columns else (df_stock.columns[2] if len(df_stock.columns) > 2 else col_stock)
    col_alert = 'status_alert' if 'status_alert' in df_stock.columns else 'status'
    col_order = 'order_rekomendasi' if 'order_rekomendasi' in df_stock.columns else 'reorder_qty'

    # Standardisasi nama kolom internal untuk tampilan UI
    df_display = df_stock.copy()
    df_display['Medicine ID'] = df_stock[col_med]
    df_display['Stok Sekarang'] = pd.to_numeric(df_stock[col_stock], errors='coerce').fillna(0).astype(int)
    df_display['Prediksi Permintaan (LSTM)'] = pd.to_numeric(df_stock[col_forecast], errors='coerce').fillna(0).astype(int)
    
    # Generate status alert otomatis jika kolom alert bawaan csv belum ada/berbeda
    if col_alert not in df_stock.columns:
        status_list = []
        order_list = []
        for idx, row in df_display.iterrows():
            stok = row['Stok Sekarang']
            pred = row['Prediksi Permintaan (LSTM)']
            safety_stock = 15
            
            if stok == 0:
                status_list.append('STOCKOUT')
                order_list.append(pred + safety_stock)
            elif stok < (pred * 0.3):
                status_list.append('KRITIS')
                order_list.append(max(0, (pred + safety_stock) - stok))
            elif stok < pred:
                status_list.append('PERLU ORDER')
                order_list.append(max(0, (pred + safety_stock) - stok))
            else:
                status_list.append('AMAN')
                order_list.append(0)
        df_display['Status Alert'] = status_list
        df_display['Rekomendasi Order'] = order_list
    else:
        df_display['Status Alert'] = df_stock[col_alert].str.upper()
        df_display['Rekomendasi Order'] = pd.to_numeric(df_stock[col_order], errors='coerce').fillna(0).astype(int)

    # ============================================================
    # BAGIAN 1: RINGKASAN METRIK (KPI)
    # ============================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Jenis Obat", value=len(df_display))

    with col2:
        stockout_count = len(df_display[df_display['Status Alert'] == 'STOCKOUT'])
        st.metric(label="🚨 Stockout (Habis)", value=stockout_count, delta=f"{stockout_count} Obat", delta_color="inverse")

    with col3:
        kritis_count = len(df_display[df_display['Status Alert'].isin(['KRITIS', 'CRITICAL'])])
        st.metric(label="🟠 Kondisi Kritis", value=kritis_count, delta=f"{kritis_count} Obat", delta_color="inverse")

    with col4:
        aman_count = len(df_display[df_display['Status Alert'] == 'AMAN'])
        st.metric(label="🟢 Status Aman", value=aman_count)

    st.markdown("---")

    # ============================================================
    # BAGIAN 2: DATASET UTAMA & VISUALISASI
    # ============================================================
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📋 Status Inventaris Tabel")
        
        def color_status(val):
            if val == 'STOCKOUT': return 'background-color: #ffcccc; color: black'
            elif val in ['KRITIS', 'CRITICAL']: return 'background-color: #ffe5cc; color: black'
            elif val in ['PERLU ORDER', 'REORDER']: return 'background-color: #ffffcc; color: black'
            return 'background-color: #e5ffcc; color: black'
            
        show_cols = ['Medicine ID', 'Stok Sekarang', 'Prediksi Permintaan (LSTM)', 'Status Alert', 'Rekomendasi Order']
        st.dataframe(df_display[show_cols].style.map(color_status, subset=['Status Alert']), use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("🔍 Deteksi Analisis Per Obat")
        selected_med = st.selectbox("Pilih ID Obat untuk melihat grafik perbandingan:", df_display['Medicine ID'].unique())
        
        med_data = df_display[df_display['Medicine ID'] == selected_med].iloc[0]
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        categories = ['Stok Saat Ini', 'Prediksi Kebutuhan']
        values = [med_data['Stok Sekarang'], med_data['Prediksi Permintaan (LSTM)']]
        colors = ['#4F8BF9', '#FF4B4B']
        
        ax.bar(categories, values, color=colors, width=0.4)
        ax.set_ylabel('Jumlah Unit')
        ax.set_title(f"Analisis Ketersediaan - {selected_med}")
        
        for i, v in enumerate(values):
            ax.text(i, v + (max(values)*0.02), str(v), ha='center', fontweight='bold')
            
        st.pyplot(fig)

    st.markdown("---")

    # ============================================================
    # BAGIAN 3: SISTEM NOTIFIKASI OTOMATIS
    # ============================================================
    st.subheader("🔔 Rekomendasi Tindakan Logistik")

    perlu_aksi = df_display[df_display['Status Alert'] != 'AMAN']

    if not perlu_aksi.empty:
        for index, row in perlu_aksi.iterrows():
            with st.expander(f"📦 {row['Medicine ID']} — Status: {row['Status Alert']}"):
                st.write(f"**Stok Saat Ini:** {row['Stok Sekarang']} unit")
                st.write(f"**Hasil Ramalan Kebutuhan (AI):** {row['Prediksi Permintaan (LSTM)']} unit")
                if row['Rekomendasi Order'] > 0:
                    st.warning(f"💡 **Rekomendasi:** Lakukan restock sebanyak **{row['Rekomendasi Order']} unit** untuk menghindari kekosongan obat.")
    else:
        st.success("✅ Seluruh stok obat aman dan mencukupi permintaan periode berikutnya.")
else:
    st.warning("Gagal memuat data. Periksa apakah file CSV Anda sudah berada di root folder repositori.")
