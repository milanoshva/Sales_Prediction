import streamlit as st
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
    <p style="font-size: 1rem;">Panduan langkah demi langkah untuk menggunakan Aplikasi Prediksi Penjualan.</p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1>📖 User Guide</h1>
    <p style="font-size: 1rem;">A step-by-step guide to using the Sales Prediction Application.</p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Mode-specific instructions
if mode == 'Normal':
    # --- NORMAL MODE GUIDE ---
    if lang == "ID":
        st.markdown("""
        <div class="info-box">
            <h3>Anda dalam Mode Normal</h3>
            <p>Mode ini dirancang untuk kemudahan dan kecepatan, fokus pada hasil prediksi dan rekomendasi yang bisa langsung ditindaklanjuti.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        #### 1. Unggah Data Penjualan Anda
        - **Di mana**: Halaman **🏠 Utama**.
        - **Caranya**: Klik "Pilih file CSV" dan unggah data transaksi Anda. Aplikasi akan otomatis memprosesnya.
        - **Hasil**: Anda akan melihat ringkasan data yang berhasil diunggah.

        #### 2. Analisis Data (Opsional)
        - **Di mana**: Halaman **📊 Analytics**.
        - **Gunakan untuk**: Melihat ringkasan penjualan, produk terlaris, dan tren penjualan dari waktu ke waktu untuk memahami bisnis Anda lebih dalam.

        #### 3. Dapatkan Prediksi & Rekomendasi
        - **Di mana**: Halaman **🔮 Prediksi**.
        - **Langkah-langkah di tab `Prediksi 1 Produk`**:
            1. **Pilih Produk**: Tentukan produk yang ingin Anda lihat masa depannya.
            2. **Tentukan Periode**: Geser slider untuk memilih berapa bulan ke depan yang ingin diprediksi.
            3. **Pilih Pendekatan**: Pilih metode prediksi. Opsi "Gabungan Terbaik (Disarankan)" adalah pilihan yang paling seimbang.
            4. **Klik "Buat Prediksi"**: Tombol biru besar untuk memulai proses.
        - **Memahami Hasil**:
            - **Tabel & Grafik**: Menampilkan angka prediksi penjualan di masa depan dibandingkan data historis.
            - **Rekomendasi**: Bagian terpenting yang memberikan saran praktis, seperti kapan harus menambah stok atau kapan perlu mengadakan promosi berdasarkan hasil prediksi.
        - **Prediksi Banyak Produk**: Gunakan tab `Prediksi Banyak Produk` untuk memprediksi beberapa produk sekaligus dalam periode yang sama.
        """, unsafe_allow_html=True)
    else:
        # --- NORMAL MODE GUIDE (ENGLISH) ---
        st.markdown("""
        <div class="info-box">
            <h3>You are in Normal Mode</h3>
            <p>This mode is designed for ease of use and speed, focusing on actionable predictions and recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. Upload Your Sales Data
        - **Where**: **🏠 Home** page.
        - **How**: Click "Choose CSV files" and upload your transaction data. The app will process it automatically.
        - **Result**: You will see a summary of the successfully uploaded data.

        #### 2. Analyze Data (Optional)
        - **Where**: **📊 Analytics** page.
        - **Use it to**: View sales summaries, top-selling products, and sales trends over time to better understand your business.

        #### 3. Get Predictions & Recommendations
        - **Where**: **🔮 Prediction** page.
        - **Steps in the `Predict 1 Product` tab**:
            1. **Select Product**: Choose the product whose future you want to see.
            2. **Set Period**: Use the slider to select how many months ahead to predict.
            3. **Choose Approach**: Select a prediction method. The "Best Combination (Recommended)" option is the most balanced choice.
            4. **Click "Generate Prediction"**: The large blue button to start the process.
        - **Understanding the Results**:
            - **Table & Chart**: Shows future sales predictions compared to historical data.
            - **Recommendations**: The most important section, providing practical advice, such as when to increase stock or when to run promotions based on the prediction results.
        - **Predict Many Products**: Use the `Predict Many Products` tab to forecast multiple products at once for the same period.
        """, unsafe_allow_html=True)
else:
    # --- ADVANCED MODE GUIDE ---
    if lang == "ID":
        st.markdown("""
        <div class="info-box">
            <h3>Anda dalam Mode Lanjutan</h3>
            <p>Mode ini memberikan kontrol penuh atas parameter prediksi dan menampilkan metrik teknis untuk evaluasi model.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. Unggah & Validasi Data
        - **Di mana**: Halaman **🏠 Utama**.
        - **Caranya**: Sama seperti mode normal, unggah file CSV Anda. Mode Lanjutan akan memberikan statistik validasi data yang lebih detail jika diperlukan.

        #### 2. Analisis Mendalam
        - **Di mana**: Halaman **📊 Analytics**.
        - **Fitur Lanjutan**: Selain fitur normal, Anda dapat melihat analisis tambahan seperti heatmap korelasi dan deteksi outlier untuk analisis data yang lebih mendalam.

        #### 3. Konfigurasi & Jalankan Prediksi
        - **Di mana**: Halaman **🔮 Prediksi**.
        - **Tab `Prediksi Interaktif`**:
            1. **Pilih Parameter**: Pilih produk, horizon waktu, dan yang terpenting, **pilih model spesifik** (`Random Forest`, `XGBoost`, `LightGBM`, atau `Gabungan`).
            2. **Jalankan Prediksi**: Klik tombol "Buat Prediksi".
            3. **Analisis Hasil**: Selain prediksi dan rekomendasi, Anda bisa menganalisis grafik untuk melihat bagaimana prediksi dibandingkan dengan data historis.
        - **Tab `Prediksi Batch`**:
            - Gunakan tab ini untuk menjalankan prediksi pada banyak produk sekaligus dengan model yang Anda pilih.

        #### 4. Evaluasi Performa Model
        - **Di mana**: Di bagian bawah halaman **🔮 Prediksi**.
        - **Caranya**: Buka expander **"Lihat Performa & Konfigurasi Model"**.
        - **Gunakan untuk**: Melihat metrik teknis seperti R², MAE, dan RMSE dari proses training model terakhir. Ini membantu Anda menilai seberapa andal model yang sedang digunakan.
        """, unsafe_allow_html=True)
    else:
        # --- ADVANCED MODE GUIDE (ENGLISH) ---
        st.markdown("""
        <div class="info-box">
            <h3>You are in Advanced Mode</h3>
            <p>This mode provides full control over prediction parameters and displays technical metrics for model evaluation.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        #### 1. Upload & Validate Data
        - **Where**: **🏠 Home** page.
        - **How**: Same as normal mode, upload your CSV files. Advanced Mode will provide more detailed data validation stats if needed.

        #### 2. In-depth Analysis
        - **Where**: **📊 Analytics** page.
        - **Advanced Features**: In addition to normal features, you can view extra analyses like correlation heatmaps and outlier detection for deeper data insights.

        #### 3. Configure & Run Predictions
        - **Where**: **🔮 Prediction** page.
        - **`Interactive Prediction` Tab**:
            1. **Select Parameters**: Choose the product, time horizon, and most importantly, **select a specific model** (`Random Forest`, `XGBoost`, `LightGBM`, or `Ensemble`).
            2. **Run Prediction**: Click the "Generate Prediction" button.
            3. **Analyze Results**: Besides the prediction and recommendations, you can analyze the chart to see how the forecast compares to historical data.
        - **`Batch Prediction` Tab**:
            - Use this tab to run predictions on multiple products at once with your chosen model.

        #### 4. Evaluate Model Performance
        - **Where**: At the bottom of the **🔮 Prediction** page.
        - **How**: Expand the **"View Model Performance & Configuration"** section.
        - **Use it to**: See technical metrics like R², MAE, and RMSE from the last model training process. This helps you judge the reliability of the currently used model.
        """, unsafe_allow_html=True)
