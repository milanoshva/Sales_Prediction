import streamlit as st
import pandas as pd
import logging
from io import StringIO
from ui.styles import set_custom_ui # Import custom UI styles
from core.data_processor import process_data

# Set up logging to app.txt
logging.basicConfig(level=logging.INFO, filename='app.txt')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(page_title="Aplikasi Prediksi Penjualan UMKM", layout="wide")

# Apply custom UI
set_custom_ui()

# Initialize session state for data and mode
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'mode' not in st.session_state:
    st.session_state.mode = 'Normal'
if 'language' not in st.session_state:
    st.session_state.language = 'ID'

# Main content
def update_welcome_text():
    if st.session_state.language == "ID":
        st.markdown("""
        <h1>🏠 Selamat Datang di Aplikasi Prediksi Penjualan UMKM Kuliner</h1>
        <p style="font-size: 0.9rem;">Aplikasi ini membantu Anda merencanakan stok dan memprediksi pendapatan dengan mudah.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <h1>🏠 Welcome to Culinary SME's Sales Prediction App</h1>
        <p style="font-size: 0.9rem;">Easily plan inventory and forecast revenue based on your sales data.</p>
        """, unsafe_allow_html=True)

update_welcome_text()

# Navbar controls
col_nav1, col_nav2 = st.columns([1, 1])
with col_nav1:
    language_options = ["Indonesia", "English"]
    current_lang = "Indonesia" if st.session_state.language == "ID" else "English"
    lang = st.selectbox("Bahasa" if st.session_state.language == "ID" else "Language",
                        language_options,
                        index=language_options.index(current_lang),
                        key="lang_select",
                        help="Pilih bahasa untuk tampilan dashboard." if st.session_state.language == "ID" else "Select language for the dashboard display.")
    if st.session_state.language != ("ID" if lang == "Indonesia" else "EN"):
        st.session_state.language = "ID" if lang == "Indonesia" else "EN"
        st.rerun()
with col_nav2:
    mode_options = ["Normal", "Advanced"]
    mode = st.selectbox("Mode" if st.session_state.language == "ID" else "Mode",
                        mode_options,
                        index=mode_options.index(st.session_state.mode),
                        key="mode_selector",
                        help="Normal: Mudah digunakan untuk pemula. Advanced: Menampilkan detail teknis untuk pengguna mahir." if st.session_state.language == "ID" else "Normal: Easy to use for beginners. Advanced: Shows technical details for advanced users.")
    st.session_state.mode = mode
    logger.info(f"Mode set to: {st.session_state.mode}")

# Advanced mode badge
if st.session_state.mode == 'Advanced':
    st.markdown('<span class="advanced-badge">{}</span>'.format("Mode Lanjutan" if st.session_state.language == "ID" else "Advanced Mode"), unsafe_allow_html=True)

# Panduan penggunaan
st.subheader("📋 Cara Memulai" if st.session_state.language == "ID" else "📋 How to Get Started")
with st.container():
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    if st.session_state.mode == 'Normal':
        if st.session_state.language == "ID":
            st.markdown("""
            1. <b>Unggah Data</b>: Unggah satu atau beberapa file CSV berisi data penjualan Anda.<br>
            2. <b>Analisis Data</b>: Gunakan halaman Analisis untuk melihat tren dan statistik.<br>                      
            3. <b>Pilih Produk</b>: Di halaman Prediksi, pilih produk untuk diprediksi.<br>
            4. <b>Lihat Prediksi</b>: Dapatkan perkiraan penjualan dan saran stok.
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            1. <b>Upload Data</b>: Upload one or more CSV files containing your sales data.<br>
            2. <b>Analisis Data</b>: Use the Analytics page to view trends and statistics.<br>                        
            3. <b>Select Product</b>: On the Prediction page, choose a product to predict.<br>
            4. <b>View Predictions</b>: Get sales forecasts and inventory suggestions.
            """, unsafe_allow_html=True)
    else:
        if st.session_state.language == "ID":
            st.markdown("""
            1. <b>Unggah Data</b>: Unggah file CSV dengan kolom waktu, nama_produk, dll.<br>
            2. <b>Analisis Data</b>: Gunakan halaman Analisis untuk melihat tren dan statistik.<br>
            3. <b>Konfigurasi Prediksi</b>: Pilih model dan parameter di halaman Prediksi.<br>
            4. <b>Evaluasi</b>: Tinjau metrik seperti MAE dan RMSE untuk hasil prediksi.<br>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            1. <b>Upload Data</b>: Upload CSV files with columns like waktu, nama_produk, etc.<br>
            2. <b>Analyze Data</b>: Use the Analytics page to view trends and statistics.<br>
            3. <b>Configure Prediction</b>: Select model and parameters on the Prediction page.<br>
            4. <b>Evaluate</b>: Review metrics like MAE and RMSE for prediction results.<br>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Data upload
st.subheader("📂 Unggah Data Penjualan" if st.session_state.language == "ID" else "📂 Upload Sales Data")
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Pilih file CSV" if st.session_state.language == "ID" else "Choose CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Unggah satu atau beberapa file CSV. Setiap file harus berisi kolom: id_transaksi, waktu, nama_pembeli, nama_kasir, nama_produk, kategori_produk, harga_satuan, jumlah, harga, harga_setelah_pajak, metode_pembayaran, total_pembayaran" if st.session_state.language == "ID" else "Upload one or more CSV files. Each file must contain columns: id_transaksi, waktu, nama_pembeli, nama_kasir, nama_produk, kategori_produk, harga_satuan, jumlah, harga, harga_setelah_pajak, metode_pembayaran, total_pembayaran"
    )

    if uploaded_files:
        with st.spinner("Memproses file..." if st.session_state.language == "ID" else "Processing files..."):
            try:
                dfs = []
                
                valid_files_processed = 0
                for uploaded_file in uploaded_files:
                    logger.info(f"Reading file: {uploaded_file.name}")
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        df_read = None
                        for sep in [',', ';', '\t']:
                            try:
                                temp_df = pd.read_csv(StringIO(content), sep=sep, encoding='utf-8')
                                if 'waktu' in temp_df.columns or 'id_transaksi' in temp_df.columns:
                                    df_read = temp_df
                                    logger.info(f"Successfully read {uploaded_file.name} with separator '{sep}'.")
                                    break
                            except Exception:
                                continue
                        
                        if df_read is None:
                            st.error(f"Gagal memproses file {uploaded_file.name}. Pastikan format CSV benar.")
                            logger.error(f"Failed to parse {uploaded_file.name} with any separator.")
                            continue

                        df_processed = process_data(df_read)
                        dfs.append(df_processed)
                        valid_files_processed += 1
                        toast_message = f"File {uploaded_file.name} berhasil diproses." if st.session_state.language == "ID" else f"File {uploaded_file.name} processed successfully."
                        st.toast(toast_message, icon="✅")

                    except Exception as e:
                        st.error(f"Error memproses file {uploaded_file.name}: {e}")
                        logger.error(f"Error processing file {uploaded_file.name}: {e}")

                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['id_transaksi', 'waktu', 'nama_produk', 'tipe_pesanan'])
                    
                    # Sort by date to ensure correct chronological order before display and processing
                    combined_df = combined_df.sort_values(by='waktu').reset_index(drop=True)
                    
                    st.session_state.df = combined_df
                    
                    summary_toast_message = f"{valid_files_processed} file berhasil digabungkan. Total {len(combined_df)} baris." if st.session_state.language == "ID" else f"{valid_files_processed} files successfully combined. Total {len(combined_df)} rows."
                    st.toast(summary_toast_message, icon="🎉")
                    
                    # --- Tampilan Data Mentah ---
                    st.markdown("**{}:**".format("Data Transaksi yang Diunggah (5 Baris Pertama)" if st.session_state.language == "ID" else "Uploaded Transaction Data (First 5 Rows)"), unsafe_allow_html=True)
                    st.dataframe(combined_df.head())

                    # --- Tampilan Rangkuman Bulanan ---
                    st.markdown("**{}:**".format("Rangkuman Penjualan Bulanan" if st.session_state.language == "ID" else "Monthly Sales Summary"), unsafe_allow_html=True)
                    
                    # Create a monthly summary
                    monthly_summary = combined_df.groupby([pd.Grouper(key='waktu', freq='M'), 'nama_produk'])['jumlah'].sum().reset_index()
                    monthly_summary['tahun'] = monthly_summary['waktu'].dt.year
                    monthly_summary['bulan'] = monthly_summary['waktu'].dt.month
                    monthly_summary = monthly_summary.sort_values(by=['nama_produk', 'waktu']).reset_index(drop=True)

                    # Calculate features on the monthly summary
                    monthly_summary['penjualan_bulan_lalu_1'] = monthly_summary.groupby('nama_produk')['jumlah'].shift(1).fillna(0)
                    monthly_summary['rata_rata_3_bulan'] = monthly_summary.groupby('nama_produk')['jumlah'].rolling(window=3, min_periods=3).mean().reset_index(level=0, drop=True).fillna(0)
                    
                    # Reorder columns for better readability
                    monthly_summary = monthly_summary[['tahun', 'bulan', 'nama_produk', 'jumlah', 'penjualan_bulan_lalu_1', 'rata_rata_3_bulan']]
                    
                    # Display a summary for each year (first 6 months)
                    display_summary = monthly_summary.groupby('tahun').head(6).reset_index(drop=True)
                    st.dataframe(display_summary)

                    if st.session_state.mode == 'Advanced':
                        # Validation metrics can be displayed here as before
                        pass

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
                logger.error(f"Error during file processing: {e}")
    else:
        if not st.session_state.df.empty:
            st.markdown("**{}:**".format("Data yang Sudah Diunggah" if st.session_state.language == "ID" else "Previously Uploaded Data"), unsafe_allow_html=True)
            st.dataframe(st.session_state.df.head())
        else:
            st.markdown("""
            <div class="st-alert st-alert-info">
                ℹ️ {}
            </div>
            """.format(
                "Silakan unggah satu atau beberapa file CSV untuk memulai." if st.session_state.language == "ID" else "Please upload one or more CSV files to get started."
            ), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
