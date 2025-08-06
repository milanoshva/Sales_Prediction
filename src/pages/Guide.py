import streamlit as st
import sys
import os
from ui.styles import set_custom_ui

# Apply custom UI
set_custom_ui()

# Get language and mode from session state
lang = st.session_state.get('language', 'ID')
mode = st.session_state.get('mode', 'Normal')

# Page Header
if lang == "ID":
    st.markdown("""
    <h1>📖 Panduan Pengguna</h1>
    <p style="font-size: 0.9rem;">Panduan langkah demi langkah untuk menggunakan Aplikasi Prediksi Penjualan.</p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1>📖 User Guide</h1>
    <p style="font-size: 0.9rem;">A step-by-step guide to using the Sales Prediction Application.</p>
    """, unsafe_allow_html=True)

# Mode-specific instructions
if mode == 'Normal':
    if lang == "ID":
        st.markdown("""
        <div class="info-box">
            <h3>Panduan Mode Normal</h3>
            <p>Mode Normal dirancang untuk kemudahan penggunaan dan memberikan wawasan penjualan yang cepat.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 1. Unggah Data Penjualan
        - **Lokasi**: Halaman `Utama`.
        - **Langkah**: Klik tombol "Pilih file CSV" dan pilih satu atau beberapa file CSV yang berisi data transaksi Anda.
        - **Format File**: Pastikan file Anda memiliki kolom yang diperlukan seperti `id_transaksi`, `waktu`, `nama_produk`, `jumlah`, dan `harga`.
        - **Hasil**: Setelah diunggah, Anda akan melihat pratinjau data gabungan di halaman Utama.

        #### 2. Analisis Data
        - **Lokasi**: Halaman `Analytics`.
        - **Fitur**:
            - **Ringkasan Penjualan**: Lihat metrik utama seperti total penjualan, jumlah transaksi, dan produk terjual.
            - **Performa Produk**: Temukan 10 produk terlaris berdasarkan unit terjual dan pendapatan.
            - **Tren Penjualan**: Amati tren penjualan dari waktu ke waktu (harian, mingguan, bulanan).
        - **Filter**: Gunakan filter di bagian atas untuk menganalisis data berdasarkan periode waktu, produk, atau kategori tertentu.

        #### 3. Dapatkan Prediksi
        - **Lokasi**: Halaman `Prediction and Models`.
        - **Langkah**:
            1. **Pilih Produk**: Pilih produk yang ingin Anda prediksi dari daftar dropdown.
            2. **Pilih Periode**: Tentukan tahun dan bulan untuk prediksi.
            3. **Klik "Prediksi Sekarang"**: Model akan memberikan prediksi jumlah penjualan untuk produk dan periode yang dipilih.
        - **Hasil**: Anda akan melihat kartu hasil dengan prediksi penjualan, perkiraan pendapatan, dan rekomendasi stok.
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h3>Normal Mode Guide</h3>
            <p>Normal Mode is designed for ease of use and provides quick sales insights.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. Upload Sales Data
        - **Location**: `Home` page.
        - **Step**: Click the "Choose CSV files" button and select one or more CSV files containing your transaction data.
        - **File Format**: Ensure your files have the required columns such as `id_transaksi`, `waktu`, `nama_produk`, `jumlah`, and `harga`.
        - **Result**: After uploading, you will see a preview of the combined data on the Home page.

        #### 2. Analyze Data
        - **Location**: `Analytics` page.
        - **Features**:
            - **Sales Summary**: View key metrics like total sales, number of transactions, and products sold.
            - **Product Performance**: Discover the top 10 best-selling products by units sold and revenue.
            - **Sales Trends**: Observe sales trends over time (daily, weekly, monthly).
        - **Filters**: Use the filters at the top to analyze data for specific time periods, products, or categories.

        #### 3. Get Predictions
        - **Location**: `Prediction and Models` page.
        - **Steps**:
            1. **Select Product**: Choose the product you want to predict from the dropdown list.
            2. **Select Period**: Specify the year and month for the prediction.
            3. **Click "Predict Now"**: The model will provide a sales quantity prediction for the selected product and period.
        - **Result**: You will see a result card with the sales prediction, estimated revenue, and stock recommendations.
        """, unsafe_allow_html=True)
else: # Advanced Mode
    if lang == "ID":
        st.markdown("""
        <div class="info-box">
            <h3>Panduan Mode Lanjutan</h3>
            <p>Mode Lanjutan menyediakan alat yang lebih mendalam untuk analisis data dan konfigurasi model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. Validasi Data Mendalam
        - **Lokasi**: Halaman `Utama` dan `Analytics`.
        - **Fitur**: Setelah mengunggah data, Anda akan melihat kartu validasi data yang menunjukkan:
            - **Total Baris**: Jumlah total baris dalam data Anda.
            - **Nilai Kosong**: Jumlah nilai yang hilang.
            - **Duplikat**: Jumlah baris duplikat.
        - **Aksi**: Anda dapat menghapus baris duplikat langsung dari halaman ini.

        #### 2. Analisis Lanjutan
        - **Lokasi**: Halaman `Analytics`.
        - **Fitur**:
            - **Heatmap Korelasi**: Pahami hubungan antara berbagai variabel numerik dalam data Anda.
            - **Deteksi Outlier**: Identifikasi dan visualisasikan nilai-nilai ekstrem (outlier) dalam data penjualan Anda.

        #### 3. Training Model Kustom
        - **Lokasi**: Halaman `Prediction and Models`.
        - **Langkah**:
            1. **Pengaturan Model**: Buka expander "Pengaturan Model Lanjutan" untuk mengonfigurasi proses training.
            2. **Pilih Model**: Pilih antara `Random Forest`, `XGBoost`, `LightGBM`, atau model `Gabungan`.
            3. **Optimasi**: Aktifkan optimasi hyperparameter untuk menemukan pengaturan model terbaik secara otomatis.
            4. **Klik "Latih Model Enhanced"**: Memulai proses training model dengan konfigurasi yang Anda pilih.
        - **Hasil**: Setelah training selesai, Anda akan melihat metrik kinerja model (MAE, RMSE, R²) dan grafik pentingnya fitur.

        #### 4. Prediksi Massal
        - **Lokasi**: Halaman `Prediction and Models`.
        - **Fitur**:
            - **Aktifkan Prediksi Massal**: Centang kotak "Prediksi Massal (Semua Produk)".
            - **Pilih Jangka Waktu**: Tentukan berapa bulan ke depan yang ingin Anda prediksi.
            - **Hasil**: Dapatkan tabel prediksi untuk semua produk, lengkap dengan ringkasan total penjualan dan pendapatan, serta opsi untuk mengunduh hasilnya.
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <h3>Advanced Mode Guide</h3>
            <p>Advanced Mode provides more in-depth tools for data analysis and model configuration.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. In-Depth Data Validation
        - **Location**: `Home` and `Analytics` pages.
        - **Features**: After uploading data, you will see data validation cards showing:
            - **Total Rows**: The total number of rows in your data.
            - **Missing Values**: The count of missing values.
            - **Duplicates**: The number of duplicate rows.
        - **Action**: You can remove duplicate rows directly from this page.

        #### 2. Advanced Analysis
        - **Location**: `Analytics` page.
        - **Features**:
            - **Correlation Heatmap**: Understand the relationships between different numerical variables in your data.
            - **Outlier Detection**: Identify and visualize extreme values (outliers) in your sales data.

        #### 3. Custom Model Training
        - **Location**: `Prediction and Models` page.
        - **Steps**:
            1. **Model Settings**: Open the "Advanced Model Settings" expander to configure the training process.
            2. **Select Model**: Choose between `Random Forest`, `XGBoost`, `LightGBM`, or an `Ensemble` model.
            3. **Optimization**: Enable hyperparameter optimization to automatically find the best model settings.
            4. **Click "Train Enhanced Model"**: Start the model training process with your chosen configuration.
        - **Result**: After training is complete, you will see model performance metrics (MAE, RMSE, R²) and a feature importance chart.

        #### 4. Bulk Predictions
        - **Location**: `Prediction and Models` page.
        - **Features**:
            - **Enable Bulk Prediction**: Check the "Bulk Prediction (All Products)" box.
            - **Select Timeframe**: Specify how many months ahead you want to predict.
            - **Result**: Get a prediction table for all products, complete with a summary of total sales and revenue, and an option to download the results.
        """, unsafe_allow_html=True)