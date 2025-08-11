import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from scipy import stats
import warnings

# Import custom modules
from ui.styles import set_custom_ui, get_plotly_template
from core.data_processor import create_advanced_features

# --- Setup ---
warnings.filterwarnings('ignore')
set_custom_ui()
plotly_template = get_plotly_template()

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PREDICTION - %(levelname)s - %(message)s')

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

# --- Helper Functions ---
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

def predict_sales(model, prediction_input, artifacts, scaler):
    """
    Performs prediction on a single, fully-featured data row.
    This function now assumes feature engineering is already done.
    """
    product_name = prediction_input['nama_produk'].iloc[0]
    logging.info(f"--- Starting prediction for product: {product_name} ---")

    all_features = artifacts['all_features_before_selection']
    selected_features = artifacts['selected_features_after_selection']
    numeric_columns = artifacts['numeric_columns_to_scale']
    product_encoder = artifacts['product_encoder']
    selector = artifacts.get('kbest_selector')
    use_log_transform = artifacts.get('config', {}).get('use_log_transform', False)
    use_feature_selection = artifacts.get('config', {}).get('use_feature_selection', False)

    try:
        product_encoded = product_encoder.transform(prediction_input[['nama_produk']])
        product_df = pd.DataFrame(
            product_encoded,
            columns=product_encoder.get_feature_names_out(['nama_produk']),
            index=prediction_input.index
        )
        prediction_row_featured = prediction_input.drop(columns=['nama_produk', 'kategori_produk', 'waktu'])
        prediction_row_featured = pd.concat([prediction_row_featured, product_df], axis=1)
    except Exception as e:
        logging.error(f"Product encoding failed for {product_name}: {e}")
        return 0

    X_aligned = prediction_row_featured.reindex(columns=all_features, fill_value=0)

    cols_to_scale = [col for col in numeric_columns if col in X_aligned.columns]
    if cols_to_scale:
        X_aligned[cols_to_scale] = scaler.transform(X_aligned[cols_to_scale])
    
    X_aligned = X_aligned.fillna(0)

    if use_feature_selection and selector:
        X_selected = selector.transform(X_aligned[all_features])
        X_final = pd.DataFrame(X_selected, columns=selected_features)
    else:
        X_final = X_aligned[selected_features]

    try:
        predicted_value = model.predict(X_final)[0]
        if use_log_transform:
            predicted_value = np.expm1(predicted_value)
        final_prediction = max(0, round(predicted_value))
        return final_prediction
    except Exception as e:
        logging.error(f"Prediction failed for {product_name}: {e}")
        st.error(f"Error during prediction for {product_name}: {e}")
        return 0

def generate_recommendations(results_df, product_name=None):
    """
    Generates actionable recommendations based on prediction results.
    """
    if results_df.empty:
        return ""

    recommendations = []
    
    if product_name:
        pred_col = 'Prediksi Penjualan'
        highest_month = results_df.loc[results_df[pred_col].idxmax()]
        lowest_month = results_df.loc[results_df[pred_col].idxmin()]
        average_sales = results_df[pred_col].mean()
        
        recommendations.append(f"#### 💡 Rekomendasi untuk {product_name}")
        
        recommendations.append(
            f"**📈 Puncak Penjualan:** Persiapkan stok lebih untuk **{highest_month['Periode']}** "
            f"karena penjualan diprediksi mencapai **{int(highest_month[pred_col])} unit**, titik tertinggi dalam periode ini."
        )
        
        if highest_month['Periode'] != lowest_month['Periode']:
            recommendations.append(
                f"**📉 Potensi Penurunan:** Waspadai penjualan di bulan **{lowest_month['Periode']}** "
                f"yang diprediksi hanya **{int(lowest_month[pred_col])} unit**. Pertimbangkan untuk membuat program promosi."
            )
        
        x = np.arange(len(results_df))
        y = results_df[pred_col].values
        try:
            slope, _, _, _, _ = stats.linregress(x, y)
            if slope > 0.5:
                trend = "menunjukkan **tren naik** yang kuat"
            elif slope > 0.1:
                trend = "menunjukkan **tren naik** yang moderat"
            elif slope < -0.5:
                trend = "menunjukkan **tren turun** yang kuat"
            elif slope < -0.1:
                trend = "menunjukkan **tren turun** yang moderat"
            else:
                trend = "cenderung **stabil**"
        except:
            trend = "tidak dapat ditentukan"
            
        recommendations.append(
            f"**📊 Analisis Tren:** Secara keseluruhan, prediksi penjualan untuk {len(results_df)} bulan ke depan {trend} "
            f"dengan rata-rata **{average_sales:.1f} unit** per bulan."
        )

    else: # Batch prediction
        pred_col = 'prediksi_penjualan'
        avg_sales_per_product = results_df.groupby('nama_produk')[pred_col].mean().sort_values(ascending=False)
        
        recommendations.append("#### 💡 Rekomendasi Hasil Prediksi Batch")
        
        if not avg_sales_per_product.empty:
            best_performer = avg_sales_per_product.index[0]
            worst_performer = avg_sales_per_product.index[-1]

            recommendations.append(
                f"**⭐ Produk Unggulan:** **{best_performer}** diprediksi menjadi produk terlaris dengan rata-rata penjualan **{int(avg_sales_per_product.iloc[0])} unit/bulan**."
            )
            if best_performer != worst_performer:
                recommendations.append(
                    f"**⚠️ Perlu Perhatian:** **{worst_performer}** memiliki prediksi penjualan terendah. Evaluasi strategi pemasaran untuk produk ini."
                )
        
        total_sales = results_df[pred_col].sum()
        recommendations.append(
            f"**💰 Potensi Penjualan:** Total prediksi penjualan dari semua produk terpilih adalah **{int(total_sales)} unit** selama periode ini."
        )

    return "\n\n".join(recommendations)

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

    # --- UI Mode Selection ---
    # The UI mode is now controlled from Home.py
    ui_mode = st.session_state.get('mode', 'Normal')
    
    if ui_mode == 'Lanjutan':
        st.markdown('<span class="advanced-badge">{}</span>'.format("Mode Lanjutan Aktif"), unsafe_allow_html=True)
    st.markdown("---")

    # --- Model Name Mapping for Normal Mode ---
    model_name_mapping = {
        'Gabungan': 'Gabungan Terbaik (Disarankan)',
        'XGBoost': 'Akurasi Tertinggi',
        'LightGBM': 'Model Cepat & Efisien',
        'Random Forest': 'Model Stabil'
    }
    reversed_model_mapping = {v: k for k, v in model_name_mapping.items()}

    # --- UI TABS ---
    tab1_label = "Prediksi 1 Produk" if ui_mode == 'Normal' else "🔮 Prediksi Interaktif"
    tab2_label = "Prediksi Banyak Produk" if ui_mode == 'Normal' else "📦 Prediksi Batch"
    tab1, tab2 = st.tabs([tab1_label, tab2_label])

    # --- TAB 1: INTERACTIVE PREDICTION ---
    with tab1:
        st.header(tab1_label)
        if ui_mode == 'Normal':
            st.markdown("Pilih satu produk untuk melihat perkiraan penjualannya di masa depan.")
        
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                product_list = sorted(df_raw['nama_produk'].unique())
                selected_product = st.selectbox("Pilih Produk", product_list, key='interactive_product')
            with col2:
                prediction_months = st.slider(
                    "Prediksi untuk berapa bulan ke depan?", 
                    1, 12, 3, 
                    key='interactive_months',
                    help="Gunakan slider ini untuk menentukan periode prediksi." if ui_mode == 'Normal' else "Set the number of months for the prediction horizon."
                )

            if ui_mode == 'Lanjutan':
                model_list = list(models.keys())
                selected_model_name_advanced = st.selectbox("Pilih Model", model_list, index=len(model_list)-1, key='interactive_model_adv')
            else:
                normal_model_options = list(model_name_mapping.values())
                selected_model_name_normal = st.selectbox("Pilih Pendekatan Prediksi", normal_model_options, index=0, key='interactive_model_normal')

        if st.button("🚀 Buat Prediksi", type="primary", key='interactive_button', use_container_width=True):
            if ui_mode == 'Lanjutan':
                selected_model_name = selected_model_name_advanced
            else:
                selected_model_name = reversed_model_mapping[selected_model_name_normal]

            with st.spinner(f"Menghitung prediksi untuk {selected_product}..."):
                try:
                    feature_maps = artifacts['feature_maps']
                    product_history = df_agg[df_agg['nama_produk'] == selected_product].copy()
                    prediction_results = []
                    last_known_date = product_history['waktu'].max() if not product_history.empty else df_raw['waktu'].max()

                    for i in range(1, prediction_months + 1):
                        prediction_date = last_known_date + relativedelta(months=i)
                        pred_input_row = pd.DataFrame([{'waktu': prediction_date, 'nama_produk': selected_product, 'jumlah': 0, 'harga': 0, 'harga_satuan': latest_prices.get(selected_product, 0), 'kategori_produk': category_map.get(selected_product, 'N/A'), 'tahun': prediction_date.year, 'bulan': prediction_date.month}])
                        combined_df = pd.concat([product_history, pred_input_row], ignore_index=True)
                        enhanced_df, _ = create_advanced_features(combined_df, feature_maps, is_training=False)
                        prediction_row_featured = enhanced_df.iloc[-1:].copy()
                        
                        model = models[selected_model_name]
                        predicted_value = predict_sales(model, prediction_row_featured, artifacts, scaler)
                        prediction_results.append({'Periode': prediction_date.strftime("%Y-%m"), 'Prediksi Penjualan': predicted_value})
                        
                        new_history_row = pred_input_row.copy()
                        new_history_row['jumlah'] = predicted_value
                        product_history = pd.concat([product_history, new_history_row], ignore_index=True)

                    st.success(f"✅ Prediksi berhasil dibuat!")
                    results_df = pd.DataFrame(prediction_results)
                    
                    st.subheader("Hasil Prediksi Penjualan")
                    st.dataframe(results_df, use_container_width=True)

                    st.subheader("Grafik Prediksi vs Historis")
                    fig = go.Figure()
                    hist_prod_df = df_agg[df_agg['nama_produk'] == selected_product]
                    fig.add_trace(go.Scatter(x=hist_prod_df['waktu'], y=hist_prod_df['jumlah'], mode='lines', name='Penjualan Historis'))
                    fig.add_trace(go.Scatter(x=results_df['Periode'], y=results_df['Prediksi Penjualan'], mode='lines+markers', name='Prediksi Penjualan', line=dict(color='red', dash='dash')))
                    fig.update_layout(title=f'Prediksi untuk {selected_product}', template=plotly_template, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    recommendation_text = generate_recommendations(results_df, selected_product)
                    st.markdown(recommendation_text)

                except Exception as e:
                    st.error(f"Gagal membuat prediksi: {e}")
                    st.exception(e)

    # --- TAB 2: BATCH PREDICTION ---
    with tab2:
        st.header(tab2_label)
        if ui_mode == 'Normal':
            st.markdown("Pilih beberapa produk sekaligus untuk melihat perkiraan penjualannya pada periode waktu yang sama.")

        with st.container(border=True):
            all_products = sorted(df_raw['nama_produk'].unique())
            selected_products_batch = st.multiselect("Pilih Produk (bisa lebih dari satu)", all_products, default=all_products[:3], key='batch_products')
            
            col1_batch, col2_batch = st.columns(2)
            with col1_batch:
                prediction_start_date = st.date_input("Mulai prediksi dari tanggal", datetime.now(), help="Pilih bulan dan tahun untuk memulai prediksi batch.")
            with col2_batch:
                months_to_predict_batch = st.number_input("Prediksi untuk berapa bulan ke depan?", min_value=1, max_value=24, value=3, key='batch_months')

            if ui_mode == 'Lanjutan':
                model_list_batch = list(models.keys())
                selected_model_batch_adv = st.selectbox("Pilih Model", model_list_batch, index=len(model_list_batch)-1, key='batch_model_adv')
            else:
                normal_model_options = list(model_name_mapping.values())
                selected_model_batch_normal = st.selectbox("Pilih Pendekatan Prediksi", normal_model_options, index=0, key='batch_model_normal')

        if st.button("🚀 Jalankan Prediksi Massal", type="primary", key='batch_button', use_container_width=True):
            if not selected_products_batch:
                st.warning("Silakan pilih setidaknya satu produk.")
            else:
                if ui_mode == 'Lanjutan':
                    selected_model_batch = selected_model_batch_adv
                else:
                    selected_model_batch = reversed_model_mapping[selected_model_batch_normal]

                with st.spinner(f"Memproses {len(selected_products_batch)} produk..."):
                    try:
                        feature_maps = artifacts['feature_maps']
                        batch_results = []
                        
                        for product in selected_products_batch:
                            product_history = df_agg[df_agg['nama_produk'] == product].copy()
                            
                            for i in range(months_to_predict_batch):
                                prediction_date = prediction_start_date + relativedelta(months=i)
                                pred_input_row = pd.DataFrame([{'waktu': prediction_date, 'nama_produk': product, 'jumlah': 0, 'harga': 0, 'harga_satuan': latest_prices.get(product, 0), 'kategori_produk': category_map.get(product, 'N/A'), 'tahun': prediction_date.year, 'bulan': prediction_date.month}])
                                combined_df = pd.concat([product_history, pred_input_row], ignore_index=True)
                                enhanced_df, _ = create_advanced_features(combined_df, feature_maps, is_training=False)
                                prediction_row_featured = enhanced_df.iloc[-1:].copy()

                                model = models[selected_model_batch]
                                predicted_value = predict_sales(model, prediction_row_featured, artifacts, scaler)
                                
                                batch_results.append({'nama_produk': product, 'tahun': prediction_date.year, 'bulan': prediction_date.strftime('%Y-%m'), 'prediksi_penjualan': predicted_value})

                        st.success("✅ Prediksi massal selesai!")
                        results_df = pd.DataFrame(batch_results)
                        
                        st.subheader("Hasil Prediksi Penjualan Massal")
                        st.dataframe(results_df, use_container_width=True)
                        
                        st.markdown("---")
                        recommendation_text = generate_recommendations(results_df)
                        st.markdown(recommendation_text)

                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil (.csv)", csv_results, f'hasil_prediksi_{datetime.now().strftime("%Y%m%d")}.csv', 'text/csv', key='download_batch', use_container_width=True)

                    except Exception as e:
                        st.error(f"Gagal membuat prediksi batch: {e}")
                        st.exception(e)

    # --- Tampilkan Performa Model ---
    if ui_mode == 'Lanjutan':
        with st.expander("ℹ️ Lihat Performa & Konfigurasi Model (Untuk Pengguna Ahli)"):
            try:
                training_results = artifacts['training_results']
                config = artifacts['config']
                timestamp = artifacts['training_timestamp']
                
                st.info(f"Hasil dari pelatihan pada: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}")
                perf_df = pd.DataFrame(training_results).T
                st.dataframe(perf_df.style.format("{:.2f}"))
                st.markdown("**Konfigurasi Pelatihan:**")
                st.json(config)
            except KeyError:
                st.warning("Info performa tidak ditemukan. Latih ulang model.")
            except Exception as e:
                st.error(f"Gagal memuat info performa: {e}")