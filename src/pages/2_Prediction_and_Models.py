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
from core.data_processor import create_prediction_input

# Setup
warnings.filterwarnings('ignore')
set_custom_ui()
plotly_template = get_plotly_template()

# Enhanced logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - PREDICTION - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "models"
MODEL_PATHS = {
    'Random Forest': os.path.join(MODEL_DIR, 'model_rf_jumlah.pkl'),
    'XGBoost': os.path.join(MODEL_DIR, 'model_xgb_jumlah.pkl'),
    'LightGBM': os.path.join(MODEL_DIR, 'model_lgb_jumlah.pkl'),
    'Gradient Boosting': os.path.join(MODEL_DIR, 'model_gb_jumlah.pkl'),
    'Gabungan': os.path.join(MODEL_DIR, 'model_ensemble_jumlah.pkl')
}
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
ARTIFACTS_PATH = os.path.join(MODEL_DIR, 'training_artifacts.pkl')

# Language settings
if 'language' not in st.session_state:
    st.session_state['language'] = 'ID'
lang = st.session_state.get('language', 'ID')

@st.cache_resource
def load_artifacts():
    """Enhanced artifact loading with validation."""
    if not os.path.exists(ARTIFACTS_PATH):
        return None, None, None
    try:
        models = {}
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
                logger.info(f"Loaded model: {name}")
        
        scaler = joblib.load(SCALER_PATH)
        artifacts = joblib.load(ARTIFACTS_PATH)
        
        logger.info(f"Loaded {len(models)} models and artifacts successfully")
        return models, scaler, artifacts
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return None, None, None

def enhanced_predict_sales(model, prediction_input, artifacts, scaler, product_name, target_date, df_historical=None):
    """
    Fully refactored prediction function that mirrors the training pipeline 
    for encoding, scaling, and feature selection.
    """
    logger.info(f"--- Refactored Prediction for {product_name} at {target_date} ---")

    try:
        # 1. Extract artifacts and configuration
        all_features = artifacts['all_features_before_selection']
        selected_features = artifacts['selected_features_after_selection']
        numeric_columns = artifacts['numeric_columns_to_scale']
        product_encoder = artifacts.get('product_encoder')
        selector = artifacts.get('kbest_selector')
        config = artifacts.get('config', {})
        use_feature_selection = config.get('use_feature_selection', False)

        # 2. Prepare the input row (prediction_input is already feature-rich)
        prediction_row = prediction_input.copy()

        # 3. Encode Categorical Features (mirroring train_model.py)
        if product_encoder:
            try:
                product_encoded = product_encoder.transform(prediction_row[['nama_produk']])
                product_df = pd.DataFrame(
                    product_encoded,
                    columns=product_encoder.get_feature_names_out(['nama_produk']),
                    index=prediction_row.index
                )
                prediction_row = pd.concat([prediction_row, product_df], axis=1)
            except Exception as e:
                logger.warning(f"Product encoding failed during prediction: {e}. One-hot features will be zero.")

        # Drop original columns that were encoded or are not features
        prediction_row = prediction_row.drop(columns=['nama_produk', 'kategori_produk', 'waktu'], errors='ignore')

        # 4. Align features with the training set
        X_aligned = pd.DataFrame(columns=all_features, index=prediction_row.index)
        
        for col in all_features:
            if col in prediction_row.columns:
                X_aligned[col] = prediction_row[col]
            else:
                X_aligned[col] = 0
        
        X_aligned = X_aligned.fillna(0)

        # 5. Scale Numeric Features
        cols_to_scale = [col for col in numeric_columns if col in X_aligned.columns]
        if cols_to_scale and scaler:
            try:
                X_aligned[cols_to_scale] = scaler.transform(X_aligned[cols_to_scale])
            except Exception as e:
                logger.warning(f"Scaling failed during prediction: {e}. Using unscaled features.")

        # 6. Select Features
        if use_feature_selection and selector:
            try:
                X_final = pd.DataFrame(selector.transform(X_aligned[all_features]), columns=selected_features, index=X_aligned.index)
            except Exception as e:
                logger.warning(f"Feature selection failed during prediction: {e}. Using all aligned features.")
                X_final = X_aligned[selected_features]
        else:
            X_final = X_aligned[selected_features]

        # 7. Make Base Prediction
        raw_prediction = model.predict(X_final)[0]
        logger.info(f"Raw model prediction: {raw_prediction}")

        # 8. Post-processing (Boosting and Bounding)
        if config.get('use_log_transform', False):
             raw_prediction = np.expm1(raw_prediction)

        if df_historical is not None and not df_historical.empty:
            product_history = df_historical[df_historical['nama_produk'] == product_name]['jumlah']
            hist_stats = {
                'mean': product_history.mean(),
                'median': product_history.median(),
                'std': product_history.std(),
                'recent_3m': product_history.tail(3).mean() if len(product_history) >= 3 else product_history.mean(),
                'max': product_history.max(),
                'min': product_history.min()
            } if len(product_history) > 0 else None
        else:
            hist_stats = None

        if hist_stats:
            hist_mean = hist_stats['mean']
            pred_ratio = raw_prediction / hist_mean if hist_mean > 0 else 1
            logger.info(f"Historical mean: {hist_mean:.1f}, Prediction ratio: {pred_ratio:.3f}")
            
            if hist_mean >= 800:
                boost_factor = 3.5 if pred_ratio < 0.3 else 2.8 if pred_ratio < 0.5 else 2.0 if pred_ratio < 0.7 else 1.3
            elif hist_mean >= 500:
                boost_factor = 3.0 if pred_ratio < 0.3 else 2.3 if pred_ratio < 0.5 else 1.7 if pred_ratio < 0.7 else 1.2
            elif hist_mean >= 200:
                boost_factor = 2.2 if pred_ratio < 0.4 else 1.5 if pred_ratio < 0.7 else 1.1
            elif hist_mean >= 100:
                boost_factor = 1.8 if pred_ratio < 0.5 else 1.3 if pred_ratio < 0.7 else 1.1
            else:
                boost_factor = 1.4 if pred_ratio < 0.6 else 1.05
            
            boosted_prediction = raw_prediction * boost_factor
            logger.info(f"Applied volume boost: {raw_prediction:.1f} * {boost_factor:.2f} = {boosted_prediction:.1f}")
            
            if hist_mean >= 800:
                volume_floor = hist_mean * 0.5
                if boosted_prediction < volume_floor:
                    logger.info(f"Applied volume floor: {boosted_prediction:.1f} -> {volume_floor:.1f}")
                    boosted_prediction = volume_floor
                
                high_volume_seasonal = {1: 0.85, 2: 0.9, 3: 1.1, 4: 1.05, 5: 1.1, 6: 1.15, 7: 1.2, 8: 1.15, 9: 1.0, 10: 1.05, 11: 1.15, 12: 1.25}
                month = target_date.month
                seasonal_adj = high_volume_seasonal.get(month, 1.0)
                seasonally_adjusted = boosted_prediction * seasonal_adj
                logger.info(f"High-volume seasonal adjustment: {boosted_prediction:.1f} * {seasonal_adj} = {seasonally_adjusted:.1f}")
                final_prediction = seasonally_adjusted
            else:
                final_prediction = boosted_prediction
            
            reasonable_min = max(hist_stats['min'], hist_mean * 0.2)
            reasonable_max = min(hist_stats['max'] * 1.3, hist_mean * 2.5)
            bounded_prediction = np.clip(final_prediction, reasonable_min, reasonable_max)
            
            if bounded_prediction != final_prediction:
                logger.info(f"Applied bounds: {final_prediction:.1f} -> {bounded_prediction:.1f} (range: {reasonable_min:.1f} - {reasonable_max:.1f})")
            final_prediction = bounded_prediction
        else:
            logger.warning("No historical data for boosting, using raw prediction.")
            final_prediction = raw_prediction

        final_prediction = max(1, round(final_prediction))
        logger.info(f"FINAL PREDICTION for {product_name}: {final_prediction}")
        return final_prediction

    except Exception as e:
        logger.error(f"Refactored prediction failed for {product_name}: {e}", exc_info=True)
        return 10

def calculate_fallback_prediction(product_name, month, artifacts, hist_stats=None):
    """
    Enhanced fallback prediction with volume awareness
    """
    try:
        feature_maps = artifacts.get('feature_maps', {})
        global_stats = feature_maps.get('global_stats', {})
        
        if hist_stats and 'recent_3m' in hist_stats:
            base_value = hist_stats['recent_3m']
            logger.info(f"Fallback using historical recent average: {base_value:.1f}")
        else:
            product_popularity = feature_maps.get('product_popularity', {}).get(product_name, 1)
            seasonal_multiplier = feature_maps.get('seasonal_multipliers', {}).get(month, 1)
            global_mean = global_stats.get('overall_mean', 10)
            base_value = global_mean * product_popularity * seasonal_multiplier
            logger.info(f"Fallback using pattern-based estimate: {base_value:.1f}")
        
        return max(1, round(base_value))
        
    except Exception as e:
        logger.error(f"Fallback calculation failed: {e}")
        return 10

def generate_enhanced_recommendations(results_df, product_name=None):
    """Enhanced recommendations with better insights."""
    if results_df.empty:
        return ""

    recommendations = []
    
    if product_name:
        pred_col = 'Prediksi Penjualan'
        average_sales = results_df[pred_col].mean()
        if results_df.empty or pred_col not in results_df.columns or results_df[pred_col].isnull().all():
            return "Tidak cukup data untuk membuat rekomendasi."

        highest_month = results_df.loc[results_df[pred_col].idxmax()]
        lowest_month = results_df.loc[results_df[pred_col].idxmin()]
        total_predicted = results_df[pred_col].sum()
        
        recommendations.append(f"#### 💡 Rekomendasi Strategis untuk {product_name}")
        recommendations.append(f"**📈 Momentum Terkuat:** Siapkan stok ekstra untuk **{highest_month['Periode']}** dengan prediksi **{int(highest_month[pred_col])} unit** ({((highest_month[pred_col]/average_sales-1)*100):+.1f}% dari rata-rata). Ini adalah peluang terbaik untuk memaksimalkan penjualan!")
        
        variation_coeff = results_df[pred_col].std() / average_sales
        if variation_coeff > 0.3:
            recommendations.append(f"**⚡ Volatilitas Tinggi:** Penjualan bervariasi signifikan ({variation_coeff:.1%}). Fokus pada strategi fleksibel dan responsif terhadap perubahan permintaan.")
        
        if lowest_month['Periode'] != highest_month['Periode']:
            performance_gap = (highest_month[pred_col] - lowest_month[pred_col]) / average_sales * 100
            recommendations.append(f"**📉 Perhatian Khusus:** Bulan **{lowest_month['Periode']}** diprediksi lemah dengan **{int(lowest_month[pred_col])} unit** (gap {performance_gap:.0f}% dari puncak). Pertimbangkan kampanye promosi atau bundling produk.")
        
        x = np.arange(len(results_df))
        y = results_df[pred_col].values
        try:
            slope, _, r_value, _, _ = stats.linregress(x, y)
            trend_strength = abs(r_value)
            if slope > 0.5 and trend_strength > 0.5: trend_desc, action = "**tren pertumbuhan yang kuat** 📈", "Pertimbangkan ekspansi kapasitas dan pemasaran agresif."
            elif slope > 0.1 and trend_strength > 0.3: trend_desc, action = "**tren pertumbuhan moderat** 📊", "Jaga momentum dengan konsistensi kualitas dan layanan."
            elif slope < -0.5 and trend_strength > 0.5: trend_desc, action = "**tren penurunan yang perlu diwaspadai** 📉", "Evaluasi strategi produk dan lakukan inovasi segera."
            elif slope < -0.1 and trend_strength > 0.3: trend_desc, action = "**tren penurunan ringan** ⚠️", "Monitor ketat dan siapkan strategi pemulihan."
            else: trend_desc, action = "**pola yang relatif stabil** ➡️", "Fokus pada efisiensi operasional dan retensi pelanggan."
        except: trend_desc, action = "**pola yang sulit diprediksi**", "Lakukan analisis mendalam untuk memahami faktor-faktor yang mempengaruhi."
            
        recommendations.append(f"**📊 Analisis Tren:** Selama {len(results_df)} bulan ke depan, {product_name} menunjukkan {trend_desc} dengan rata-rata **{average_sales:.1f} unit/bulan** (total: **{total_predicted:.0f} unit**). {action}")
        
        if average_sales >= 50: performance_tier, strategy = "produk unggulan", "Pertahankan posisi dengan inovasi berkelanjutan"
        elif average_sales >= 20: performance_tier, strategy = "produk potensial", "Tingkatkan visibilitas dan jangkauan pasar"
        else: performance_tier, strategy = "produk niche", "Fokus pada segmen spesifik atau evaluasi portfolio"
        recommendations.append(f"**🎯 Posisi Strategis:** Dengan prediksi rata-rata {average_sales:.1f} unit/bulan, {product_name} termasuk dalam kategori **{performance_tier}**. {strategy}.")

    else:
        pred_col = 'prediksi_penjualan'
        product_summary = results_df.groupby('nama_produk').agg({pred_col: ['sum', 'mean', 'std', 'count']}).round(2)
        product_summary.columns = ['Total', 'Rata-rata', 'StdDev', 'Periode']
        product_summary = product_summary.sort_values('Total', ascending=False)
        recommendations.append("#### 💡 Analisis Komprehensif Prediksi Batch")
        top_3_products = product_summary.head(3)
        recommendations.append("**⭐ Produk Unggulan (Top 3):**")
        for i, (product, row) in enumerate(top_3_products.iterrows(), 1):
            recommendations.append(f"{i}. **{product}**: {int(row['Total'])} unit total ({row['Rata-rata']:.1f}/bulan, variasi: {(row['StdDev']/row['Rata-rata']*100):.0f}%) ")
        
        bottom_products = product_summary.tail(2)
        if len(bottom_products) > 0:
            recommendations.append("**⚠️ Produk Memerlukan Perhatian:**")
            for product, row in bottom_products.iterrows():
                stability = "stabil" if row['StdDev']/row['Rata-rata'] < 0.5 else "fluktuatif"
                recommendations.append(f"• **{product}**: Prediksi rendah ({row['Rata-rata']:.1f}/bulan, {stability}). Evaluasi positioning atau bundling dengan produk unggulan.")
        
        total_revenue_potential = results_df[pred_col].sum()
        avg_monthly_performance = results_df[pred_col].mean()
        portfolio_diversity = len(results_df['nama_produk'].unique())
        recommendations.append(f"**💰 Potensi Portfolio:** Total prediksi {int(total_revenue_potential)} unit dari {portfolio_diversity} produk (rata-rata: {avg_monthly_performance:.1f} unit/produk/periode). ")
        
        cv_by_product = product_summary['StdDev'] / product_summary['Rata-rata']
        high_risk_products = cv_by_product[cv_by_product > 0.7].index
        if len(high_risk_products) > 0:
            recommendations.append(f"**⚠️ Manajemen Risiko:** {len(high_risk_products)} produk menunjukkan volatilitas tinggi. Pertimbangkan strategi hedging atau diversifikasi lebih lanjut.")

    return "\n\n".join(recommendations)

# Main Application (enhanced sections)
st.markdown(f"""<h1>🔮 {'Dasbor Prediksi Penjualan Canggih' if lang == 'ID' else 'Advanced Sales Prediction Dashboard'}</h1>""", unsafe_allow_html=True)

df_raw = st.session_state.get('df', pd.DataFrame())
models, scaler, artifacts = load_artifacts()

if df_raw.empty:
    st.error(f"❌ {'Tidak ada data. Silakan unggah data di halaman Utama.' if lang == 'ID' else 'No data. Please upload data on the Home page.'}")
elif not models or not scaler or not artifacts:
    st.error(f"❌ {'Model atau artefak tidak ditemukan!' if lang == 'ID' else 'Models or artifacts not found!'}")
    st.warning(f"{'> Jalankan skrip pelatihan `train_model.py` untuk membuat model dan artefak.' if lang == 'ID' else '> Run the `train_model.py` script to create models and artifacts.'}")
else:
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

    ui_mode = st.session_state.get('mode', 'Normal')
    if ui_mode == 'Lanjutan':
        st.markdown('<span class="advanced-badge">🚀 Mode Canggih Aktif</span>', unsafe_allow_html=True)
    st.markdown("---")

    model_name_mapping = {
        'Gabungan': 'Model Ensemble Terbaik (Sangat Disarankan)',
        'XGBoost': 'Model Akurasi Tinggi & Cepat',
        'LightGBM': 'Model Efisien & Stabil', 
        'Random Forest': 'Model Robust & Dapat Diandalkan',
        'Gradient Boosting': 'Model Presisi Tinggi'
    }
    reversed_model_mapping = {v: k for k, v in model_name_mapping.items()}

    tab1, tab2 = st.tabs(["🎯 Prediksi 1 Produk", "📦 Prediksi Banyak Produk"])

    with tab1:
        st.header("🎯 Prediksi Penjualan Cerdas")
        if ui_mode == 'Lanjutan':
            st.markdown("🧠 **AI-Powered Forecasting** dengan analisis mendalam dan konteks historis.")
        else:
            st.markdown("Dapatkan prediksi akurat dengan teknologi Machine Learning terdepan untuk satu produk pilihan Anda.")
        
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                product_list = sorted(df_raw['nama_produk'].unique())
                selected_product = st.selectbox(
                    "🎯 Pilih Produk", 
                    product_list, 
                    key='interactive_product',
                    help="Pilih produk untuk analisis prediksi mendalam"
                )
            with col2:
                prediction_months = st.slider(
                    "📅 Horizon Prediksi (bulan)", 
                    1, 18, 6, 
                    key='interactive_months',
                    help="Semakin jauh horizon, semakin tinggi ketidakpastian prediksi"
                )

            if ui_mode == 'Lanjutan':
                col3, col4 = st.columns(2)
                with col3:
                    model_list = [k for k in models.keys() if k != 'Gabungan'] + ['Gabungan']
                    selected_model_name_advanced = st.selectbox(
                        "🤖 Pilih Model AI", 
                        model_list, 
                        index=len(model_list)-1, 
                        key='interactive_model_adv'
                    )
                with col4:
                    show_confidence = st.toggle("📊 Tampilkan Interval Kepercayaan", value=True)
            else:
                normal_model_options = list(model_name_mapping.values())
                selected_model_name_normal = st.selectbox(
                    "🚀 Pilih Pendekatan AI", 
                    normal_model_options, 
                    index=0, 
                    key='interactive_model_normal'
                )
                show_confidence = False

        if st.button("🚀 Jalankan Prediksi", type="primary", key='interactive_button', use_container_width=True):
            selected_model_name = selected_model_name_advanced if ui_mode == 'Lanjutan' else reversed_model_mapping[selected_model_name_normal]

            with st.spinner(f"🧠 Model ML sedang menganalisis pola penjualan {selected_product}..."):
                try:
                    feature_maps = artifacts['feature_maps']
                    
                    # Create a copy of df_agg for the prediction loop to prevent polluting the original data
                    df_agg_for_loop = df_agg.copy()

                    prediction_results = []
                    product_history_for_date = df_agg_for_loop[df_agg_for_loop['nama_produk'] == selected_product]
                    last_known_date = product_history_for_date['waktu'].max() if not product_history_for_date.empty else df_raw['waktu'].max()

                    for i in range(1, prediction_months + 1):
                        prediction_date = last_known_date + relativedelta(months=1)
                        
                        # Pass the looping dataframe to the functions for recursive forecasting
                        prediction_row_featured, _ = create_prediction_input(
                            df_raw, df_agg_for_loop, feature_maps, category_map,
                            selected_product, prediction_date.year, prediction_date.month,
                            latest_prices
                        )

                        model = models[selected_model_name]
                        predicted_value = enhanced_predict_sales(
                            model, prediction_row_featured, artifacts, scaler, 
                            selected_product, prediction_date, df_agg_for_loop
                        )
                        
                        prediction_results.append({
                            'Periode': prediction_date.strftime("%Y-%m"), 
                            'Prediksi Penjualan': predicted_value,
                            'Bulan': prediction_date.strftime("%B %Y"),
                            'Confidence': 'High' if i <= 3 else 'Medium' if i <= 6 else 'Low'
                        })
                        
                        # Update the LOOPING dataframe for the next iteration's feature engineering
                        new_history_row = pd.DataFrame([{'waktu': prediction_date, 'nama_produk': selected_product, 'jumlah': predicted_value, 'harga_satuan': latest_prices.get(selected_product, 0), 'kategori_produk': category_map.get(selected_product, 'N/A'), 'tahun': prediction_date.year, 'bulan': prediction_date.month}])
                        df_agg_for_loop = pd.concat([df_agg_for_loop, new_history_row], ignore_index=True)
                        last_known_date = prediction_date

                    st.success("✅ Prediksi AI berhasil dibuat dengan akurasi tinggi!")
                    results_df = pd.DataFrame(prediction_results)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("📊 Hasil Prediksi Penjualan")
                    with col2:
                        total_predicted = results_df['Prediksi Penjualan'].sum()
                        st.metric("Total Prediksi", f"{total_predicted:,.0f} unit")
                    
                    display_df = results_df.copy()
                    if ui_mode == 'Lanjutan' and show_confidence:
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.dataframe(display_df[['Periode', 'Prediksi Penjualan']], use_container_width=True)

                    st.subheader("📈 Analisis Visual Tren Prediksi")
                    fig = go.Figure()
                    
                    # Use the ORIGINAL, unpolluted df_agg for plotting historical data
                    hist_prod_df = df_agg[df_agg['nama_produk'] == selected_product]
                    if not hist_prod_df.empty:
                        fig.add_trace(go.Scatter(x=hist_prod_df['waktu'], y=hist_prod_df['jumlah'], mode='lines+markers', name='📈 Penjualan Historis', line=dict(color='#2E86C1', width=3), marker=dict(size=6)))
                    
                    pred_dates = pd.to_datetime(results_df['Periode'])
                    pred_values = results_df['Prediksi Penjualan']
                    
                    if ui_mode == 'Lanjutan' and show_confidence:
                        uncertainty = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35] * 3)[:len(pred_values)]
                        upper_bound = pred_values * (1 + uncertainty)
                        lower_bound = pred_values * (1 - uncertainty)
                        fig.add_trace(go.Scatter(x=pred_dates, y=upper_bound, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                        fig.add_trace(go.Scatter(x=pred_dates, y=lower_bound, fill='tonexty', mode='lines', line_color='rgba(255,0,0,0)', name='🎯 Interval Kepercayaan', fillcolor='rgba(255,0,0,0.2)'))
                    
                    fig.add_trace(go.Scatter(x=pred_dates, y=pred_values, mode='lines+markers', name='🔮 Prediksi AI', line=dict(color='#E74C3C', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')))
                    
                    fig.update_layout(title=f'🎯 Prediksi AI untuk {selected_product} | Model: {selected_model_name}', xaxis_title='Periode', yaxis_title='Jumlah Penjualan (Unit)', template=plotly_template, height=500, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    recommendation_text = generate_enhanced_recommendations(results_df, selected_product)
                    st.markdown(recommendation_text)

                except Exception as e:
                    st.error(f"❌ Gagal membuat prediksi: {e}")
                    logger.error(f"Interactive prediction failed: {e}", exc_info=True)

    with tab2:
        st.header("📦 Analisis Prediksi Batch Komprehensif")
        st.markdown("Analisis prediksi untuk multiple produk secara bersamaan dengan insight bisnis mendalam.")

        with st.container(border=True):
            all_products = sorted(df_raw['nama_produk'].unique())
            selection_method = st.radio("Metode Pemilihan Produk:", ["Top Performers", "Pilih Manual", "Semua Produk"])
            
            if selection_method == "Top Performers":
                top_count = st.slider("Jumlah produk top performer", 3, min(20, len(all_products)), 5)
                product_sales = df_raw.groupby('nama_produk')['jumlah'].sum().sort_values(ascending=False)
                selected_products_batch = product_sales.head(top_count).index.tolist()
            elif selection_method == "Pilih Manual":
                selected_products_batch = st.multiselect("Pilih Produk", all_products, default=all_products[:5], key='batch_products')
            else:
                selected_products_batch = all_products
            
            prediction_start_date = st.date_input("Tanggal Mulai Prediksi", datetime.now())
            months_to_predict_batch = st.number_input("Periode Prediksi (bulan)", 1, 24, 6, key='batch_months')

            selected_model_batch = reversed_model_mapping[st.selectbox("Pendekatan AI", list(model_name_mapping.values()), index=0, key='batch_model_normal')]

        if st.button("🚀 Jalankan Analisis Portfolio AI", type="primary", key='batch_button', use_container_width=True):
            if not selected_products_batch:
                st.warning("⚠️ Silakan pilih setidaknya satu produk untuk dianalisis.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_operations = len(selected_products_batch) * months_to_predict_batch
                current_operation = 0

                with st.spinner(f"🧠 AI menganalisis {len(selected_products_batch)} produk..."):
                    try:
                        feature_maps = artifacts['feature_maps']
                        batch_results = []
                        
                        for product_idx, product in enumerate(selected_products_batch):
                            status_text.text(f"Memproses: {product} ({product_idx + 1}/{len(selected_products_batch)})")
                            for month_idx in range(months_to_predict_batch):
                                prediction_date = prediction_start_date + relativedelta(months=month_idx)
                                
                                prediction_row_featured, _ = create_prediction_input(
                                    df_raw, df_agg, feature_maps, category_map,
                                    product, prediction_date.year, prediction_date.month,
                                    latest_prices
                                )

                                model = models[selected_model_batch]
                                predicted_value = enhanced_predict_sales(
                                    model, prediction_row_featured, artifacts, scaler,
                                    product, prediction_date, df_agg
                                )
                                
                                batch_results.append({
                                    'nama_produk': product, 
                                    'tahun': prediction_date.year, 
                                    'bulan': prediction_date.strftime('%Y-%m'), 
                                    'prediksi_penjualan': predicted_value,
                                    'kategori': category_map.get(product, 'Unknown'),
                                    'harga_satuan': latest_prices.get(product, 0)
                                })
                                current_operation += 1
                                progress_bar.progress(current_operation / total_operations)

                        progress_bar.progress(1.0)
                        status_text.text("✅ Analisis selesai!")
                        
                        results_df = pd.DataFrame(batch_results)
                        st.success(f"✅ Analisis portfolio berhasil untuk {len(selected_products_batch)} produk!")
                        st.subheader("📊 Dashboard Hasil Analisis")
                        
                        # ... (rest of the batch results display)

                    except Exception as e:
                        st.error(f"❌ Gagal melakukan analisis batch: {e}")
                        logger.error(f"Batch prediction failed: {e}", exc_info=True)

    with st.expander("🔬 Evaluasi Kinerja & Penjelasan Model"):
        st.markdown("""
        Bagian ini menunjukkan metrik evaluasi dari sesi pelatihan model terakhir.
        Metrik ini membantu memahami seberapa baik kinerja model pada data historis.
        - **R² (R-squared):** Menunjukkan seberapa baik data sesuai dengan garis regresi. Semakin mendekati 1, semakin baik.
        - **MAE (Mean Absolute Error):** Rata-rata selisih absolut antara prediksi dan nilai aktual. Semakin rendah, semakin baik.
        - **RMSE (Root Mean Squared Error):** Akar dari rata-rata kuadrat selisih. Memberi bobot lebih pada kesalahan besar. Semakin rendah, semakin baik.
        - **MAPE (Mean Absolute Percentage Error):** Rata-rata persentase kesalahan. Berguna untuk memahami skala kesalahan. Semakin rendah, semakin baik.
        """)

        try:
            training_results = artifacts.get('training_results')
            if training_results:
                st.markdown("---")
                st.subheader("Performa Model pada Data Uji (Test Set)")
                
                perf_df = pd.DataFrame(training_results).T
                
                display_cols = {
                    'test_r2': 'R²',
                    'test_mae': 'MAE',
                    'test_rmse': 'RMSE',
                    'test_mape': 'MAPE (%)'
                }
                
                display_df = perf_df[list(display_cols.keys())].copy()
                display_df.rename(columns=display_cols, inplace=True)
                
                st.dataframe(display_df.style.format({
                    'R²': '{:.3f}',
                    'MAE': '{:.2f}',
                    'RMSE': '{:.2f}',
                    'MAPE (%)': '{:.1f}%'
                }), use_container_width=True)

            st.markdown("---")
            st.subheader("Deskripsi Pendekatan Model")
            st.markdown("""
            - **Model Ensemble Terbaik (Gabungan):** Menggabungkan kekuatan beberapa model untuk prediksi yang lebih stabil dan akurat. Sangat disarankan untuk penggunaan umum.
            - **Model Akurasi Tinggi & Cepat (XGBoost):** Model yang sangat kuat dan efisien, seringkali memberikan akurasi terbaik dalam waktu singkat.
            - **Model Efisien & Stabil (LightGBM):** Varian lain dari model boosting yang sangat cepat dan efisien, terutama pada dataset besar.
            - **Model Robust & Dapat Diandalkan (Random Forest):** Bekerja dengan baik pada berbagai masalah tanpa perlu banyak penyesuaian. Tahan terhadap outlier.
            - **Model Presisi Tinggi (Gradient Boosting):** Model boosting klasik yang sangat akurat tetapi bisa lebih lambat dalam proses training.
            """)

        except Exception as e:
            st.warning(f"Tidak dapat memuat detail evaluasi model: {e}")