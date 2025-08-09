import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import warnings

# Import custom modules
from ui.styles import set_custom_ui, get_plotly_template
from core.data_processor import create_advanced_features

# --- Setup ---
warnings.filterwarnings('ignore')
set_custom_ui()
plotly_template = get_plotly_template()

# --- Constants ---
MODEL_DIR = "models"
MODEL_PATHS = {
    'Random Forest': os.path.join(MODEL_DIR, 'model_rf_jumlah.pkl'),
    'XGBoost': os.path.join(MODEL_DIR, 'model_xgb_jumlah.pkl'),
    'LightGBM': os.path.join(MODEL_DIR, 'model_lgb_jumlah.pkl'),
    'Gabungan': os.path.join(MODEL_DIR, 'model_ensemble_jumlah.pkl')
}
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
ARTIFACTS_PATH = os.path.join(MODEL_DIR, 'training_artifacts.pkl')

# --- Language Settings ---
if 'language' not in st.session_state:
    st.session_state['language'] = 'ID'
lang = st.session_state.get('language', 'ID')

# --- Helper Function to Load Artifacts ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        return None, None, None
    try:
        models = {name: joblib.load(path) for name, path in MODEL_PATHS.items() if os.path.exists(path)}
        scaler = joblib.load(SCALER_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        return models, scaler, artifacts
    except Exception as e:
        st.error(f"Gagal memuat artefak model: {e}")
        return None, None, None

# --- Prediction Function ---
def predict_sales(model, data_input, artifacts, scaler):
    """
    Membuat prediksi penjualan tunggal menggunakan pipeline yang sudah dilatih.
    Fungsi ini sekarang lebih sederhana dan andal, menggunakan artefak yang disimpan.
    """
    # Ekstrak artefak yang diperlukan
    all_features = artifacts['all_features_before_selection']
    selected_features = artifacts['selected_features_after_selection']
    numeric_columns = artifacts['numeric_columns_to_scale']
    product_encoder = artifacts['product_encoder']
    selector = artifacts.get('kbest_selector')
    use_log_transform = artifacts.get('config', {}).get('use_log_transform', False)

    # Pastikan data_input adalah DataFrame
    if not isinstance(data_input, pd.DataFrame):
        data_input = pd.DataFrame([data_input])

    # 1. One-Hot Encode 'nama_produk'
    product_encoded = product_encoder.transform(data_input[['nama_produk']])
    product_df = pd.DataFrame(product_encoded, columns=product_encoder.get_feature_names_out(['nama_produk']), index=data_input.index)
    
    # Gabungkan fitur yang di-encode dan hapus kolom asli
    processed_data = pd.concat([data_input.drop(columns=['nama_produk', 'kategori_produk']), product_df], axis=1)

    # 2. Pastikan semua kolom dari training ada di data prediksi
    # Buat DataFrame dengan urutan kolom yang benar, isi dengan 0 jika tidak ada
    X_pred = pd.DataFrame(0, index=processed_data.index, columns=all_features)
    # Isi kolom yang ada dari data yang diproses
    common_cols = [col for col in all_features if col in processed_data.columns]
    X_pred[common_cols] = processed_data[common_cols]

    # 3. Terapkan Robust Scaler
    # Pastikan hanya kolom numerik yang ada yang di-scale
    cols_to_scale = [col for col in numeric_columns if col in X_pred.columns]
    if cols_to_scale:
        X_pred[cols_to_scale] = scaler.transform(X_pred[cols_to_scale])
    
    # Isi nilai NaN yang mungkin muncul setelah proses
    X_pred.fillna(0, inplace=True)

    # 4. Terapkan Feature Selection (jika digunakan saat training)
    if selector:
        X_selected = selector.transform(X_pred[all_features]) # Gunakan all_features untuk konsistensi
    else:
        # Jika tidak ada selector, gunakan fitur yang sudah dipilih saat training
        X_selected = X_pred[selected_features]

    # 5. Lakukan Prediksi
    predicted_value = model.predict(X_selected)[0]
    
    # 6. Transformasi balik jika target di-log
    if use_log_transform:
        predicted_value = np.expm1(predicted_value)
    
    # Pastikan hasil tidak negatif
    return max(0, round(predicted_value))

# --- Main App ---
st.markdown(f"""<h1>🔮 {'Dasbor Prediksi Penjualan' if lang == 'ID' else 'Sales Prediction Dashboard'}</h1>""", unsafe_allow_html=True)

df_raw = st.session_state.get('df', pd.DataFrame())
models, scaler, artifacts = load_artifacts()

if df_raw.empty:
    st.error(f"❌ {'Tidak ada data. Silakan unggah data di halaman Utama.' if lang == 'ID' else 'No data. Please upload data on the Home page.'}")
elif not models or not scaler or not artifacts:
    st.error(f"❌ {'Model atau artefak tidak ditemukan!' if lang == 'ID' else 'Models or artifacts not found!'}")
    st.warning(f"{'> Jalankan skrip `python train_model.py` untuk melatih model.' if lang == 'ID' else '> Run the `python train_model.py` script to train the models.'}")
else:
    # --- Data Preparation ---
    df_agg = df_raw.groupby(['nama_produk', pd.Grouper(key='waktu', freq='M')]).agg(
        jumlah=('jumlah', 'sum'),
        harga=('harga', 'sum'),
        harga_satuan=('harga_satuan', 'mean'),
        kategori_produk=('kategori_produk', 'first')
    ).reset_index()
    df_agg['tahun'] = df_agg['waktu'].dt.year
    df_agg['bulan'] = df_agg['waktu'].dt.month

    latest_prices = df_raw.sort_values('waktu').groupby('nama_produk')['harga_satuan'].last().to_dict()
    category_map = df_raw.groupby('nama_produk')['kategori_produk'].first().to_dict()

    # --- UI TABS ---
    tab1, tab2 = st.tabs([f"🔮 {'Prediksi Interaktif' if lang == 'ID' else 'Interactive Prediction'}", 
                          f"📦 {'Prediksi Batch' if lang == 'ID' else 'Batch Prediction'}"])

    with tab1:
        st.markdown(f"""<p style='font-size:0.9rem'>{'Pilih satu produk untuk melihat prediksi penjualannya di masa depan secara iteratif.' if lang == 'ID' else 'Select a single product to see its future sales prediction iteratively.'}</p>""", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            product_list = sorted(df_raw['nama_produk'].unique())
            selected_product = st.selectbox(f"{'Pilih Produk' if lang == 'ID' else 'Select Product'}", product_list, key='interactive_product')
        with col2:
            model_list = list(models.keys())
            selected_model_name = st.selectbox(f"{'Pilih Model' if lang == 'ID' else 'Select Model'}", model_list, index=len(model_list)-1, key='interactive_model')
        with col3:
            prediction_months = st.slider(f"{'Jumlah Bulan Prediksi' if lang == 'ID' else 'Months to Predict'}", 1, 12, 3, key='interactive_months')

        if st.button(f"🚀 {'Buat Prediksi' if lang == 'ID' else 'Generate Prediction'}", type="primary", key='interactive_button'):
            with st.spinner(f"{'Membuat prediksi untuk' if lang == 'ID' else 'Generating predictions for'} {selected_product}..."):
                try:
                    feature_maps = artifacts['feature_maps']
                    historical_df = df_agg.copy()
                    prediction_results = []
                    last_known_date = df_raw['waktu'].max()

                    for i in range(1, prediction_months + 1):
                        prediction_date = last_known_date + relativedelta(months=i)
                        pred_input_df = pd.DataFrame([{
                            'waktu': prediction_date, 
                            'nama_produk': selected_product, 
                            'jumlah': 0,
                            'harga': 0, 
                            'harga_satuan': latest_prices.get(selected_product, 0),
                            'kategori_produk': category_map.get(selected_product, 'N/A')
                        }])
                        
                        combined_for_features = pd.concat([historical_df, pred_input_df], ignore_index=True)
                        enhanced_df, _ = create_advanced_features(combined_for_features, feature_maps, is_training=False)
                        enhanced_df['kategori_encoded'] = enhanced_df['kategori_produk'].astype('category').cat.codes
                        
                        # Prepare prediction_row with all necessary features, including nama_produk for encoding
                        prediction_row_base = enhanced_df.iloc[-1:].copy()
                        
                        # The predict_sales function now handles the product_encoder internally
                        model = models[selected_model_name]
                        predicted_value = predict_sales(model, prediction_row_base, artifacts, scaler)
                        
                        prediction_results.append({'Periode': prediction_date.strftime("%Y-%m"), 'Prediksi Penjualan': predicted_value})
                        
                        new_row = pred_input_df.copy()
                        new_row['jumlah'] = predicted_value
                        historical_df = pd.concat([historical_df, new_row], ignore_index=True)

                    st.success(f"✅ {'Prediksi berhasil' if lang == 'ID' else 'Prediction successful'}")
                    results_df = pd.DataFrame(prediction_results)
                    st.dataframe(results_df, use_container_width=True)

                    fig = go.Figure()
                    hist_prod_df = df_agg[df_agg['nama_produk'] == selected_product]
                    fig.add_trace(go.Scatter(x=hist_prod_df['waktu'], y=hist_prod_df['jumlah'], mode='lines', name=f'{'Historis' if lang == 'ID' else 'Historical'}'))
                    fig.add_trace(go.Scatter(x=results_df['Periode'], y=results_df['Prediksi Penjualan'], mode='lines+markers', name=f'{'Prediksi' if lang == 'ID' else 'Prediction'}', line=dict(color='red', dash='dash')))
                    fig.update_layout(title=f'{'Prediksi untuk' if lang == 'ID' else 'Prediction for'} {selected_product}', template=plotly_template)
                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Gagal membuat prediksi: {e}")
                    st.exception(e)

    with tab2:
        st.markdown(f"""<p style='font-size:0.9rem'>{'Pilih beberapa produk dan periode untuk prediksi massal. Setiap prediksi dihitung secara independen.' if lang == 'ID' else 'Select multiple products and a period for batch prediction. Each prediction is calculated independently.'}</p>""", unsafe_allow_html=True)
        
        all_products = sorted(df_raw['nama_produk'].unique())
        selected_products_batch = st.multiselect(f"{'Pilih Produk' if lang == 'ID' else 'Select Products'}", all_products, default=all_products[:3], key='batch_products')
        
        col1_batch, col2_batch, col3_batch = st.columns(3)
        with col1_batch:
            current_year = datetime.now().year
            year_range = list(range(current_year - 2, current_year + 5))
            selected_year = st.selectbox(f"{'Pilih Tahun' if lang == 'ID' else 'Select Year'}", year_range, index=year_range.index(current_year), key='batch_year')
        with col2_batch:
            month_names = [datetime(2000, i, 1).strftime('%B') for i in range(1, 13)]
            month_map = {name: i for i, name in enumerate(month_names, 1)}
            selected_month_name = st.selectbox(f"{'Pilih Bulan' if lang == 'ID' else 'Select Month'}", month_names, index=datetime.now().month - 1, key='batch_month')
            selected_month = month_map[selected_month_name]
        with col3_batch:
            months_to_predict_batch = st.number_input(f"{'Jumlah Bulan Prediksi' if lang == 'ID' else 'Number of Months to Predict'}", min_value=1, max_value=24, value=6, key='batch_months')

        model_list_batch = list(models.keys())
        selected_model_batch = st.selectbox(f"{'Pilih Model' if lang == 'ID' else 'Select Model'}", model_list_batch, index=len(model_list_batch)-1, key='batch_model')

        if st.button(f"🚀 {'Jalankan Prediksi Batch' if lang == 'ID' else 'Run Batch Prediction'}", type="primary", key='batch_button'):
            if not selected_products_batch:
                st.warning(f"{'Silakan pilih setidaknya satu produk.' if lang == 'ID' else 'Please select at least one product.'}")
            else:
                with st.spinner(f"{'Memproses prediksi batch...' if lang == 'ID' else 'Processing batch predictions...'}"):
                    try:
                        feature_maps = artifacts['feature_maps']
                        historical_df = df_agg.copy()
                        batch_results = []
                        start_date = datetime(selected_year, selected_month, 1)

                        for product in selected_products_batch:
                            for i in range(months_to_predict_batch):
                                prediction_date = start_date + relativedelta(months=i)
                                
                                pred_input_df = pd.DataFrame([{
                                    'waktu': prediction_date, 
                                    'nama_produk': product, 
                                    'jumlah': 0,
                                    'harga': 0, 
                                    'harga_satuan': latest_prices.get(product, 0),
                                    'kategori_produk': category_map.get(product, 'N/A')
                                }])
                                
                                combined_for_features = pd.concat([historical_df, pred_input_df], ignore_index=True)
                                enhanced_df, _ = create_advanced_features(combined_for_features, feature_maps, is_training=False)
                                enhanced_df['kategori_encoded'] = enhanced_df['kategori_produk'].astype('category').cat.codes
                                
                                # Prepare prediction_row with all necessary features, including nama_produk for encoding
                                prediction_row_base = enhanced_df.iloc[-1:].copy()
                                
                                # The predict_sales function now handles the product_encoder internally
                                model = models[selected_model_batch]
                                predicted_value = predict_sales(model, prediction_row_base, artifacts, scaler)
                                
                                batch_results.append({
                                    'nama_produk': product, 
                                    'tahun': prediction_date.year,
                                    'bulan': prediction_date.month,
                                    'prediksi_penjualan': predicted_value
                                })

                        st.success(f"✅ {'Prediksi batch selesai' if lang == 'ID' else 'Batch prediction finished'}")
                        results_df = pd.DataFrame(batch_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"📥 {'Unduh Hasil' if lang == 'ID' else 'Download Results'}", csv_results, 'hasil_prediksi_batch.csv', 'text/csv', key='download_batch')

                    except Exception as e:
                        st.error(f"Gagal membuat prediksi batch: {e}")
                        st.exception(e)

    # --- Tampilkan Performa Model ---
    with st.expander(f"ℹ️ {'Lihat Performa Model' if lang == 'ID' else 'View Model Performance'}"):
        try:
            training_results = artifacts['training_results']
            config = artifacts['config']
            timestamp = artifacts['training_timestamp']
            
            st.info(f"{'*Hasil dari pelatihan pada*' if lang == 'ID' else '*Results from training on*'} {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}")
            perf_df = pd.DataFrame(training_results).T
            st.dataframe(perf_df.style.format("{:.2f}"))
            st.markdown("**Konfigurasi:**")
            st.json(config)
        except KeyError:
            st.warning("Info performa tidak ditemukan. Latih ulang model.")
        except Exception as e:
            st.error(f"Gagal memuat info performa: {e}")
