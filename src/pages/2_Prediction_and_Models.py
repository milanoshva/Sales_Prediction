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

# Get language and mode from session state
lang = st.session_state.get('language', 'ID')
mode = st.session_state.get('mode', 'Normal')

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

def enhanced_predict_sales(model, prediction_input, artifacts, scaler, product_name, target_date):
    """
    Enhanced prediction function with better handling of feature defaults and scaling.
    """
    logger.info(f"--- Enhanced prediction for {product_name} at {target_date} ---")

    try:
        # Extract configuration and artifacts
        all_features = artifacts['all_features_before_selection']
        selected_features = artifacts['selected_features_after_selection']
        numeric_columns = artifacts['numeric_columns_to_scale']
        product_encoder = artifacts['product_encoder']
        selector = artifacts.get('kbest_selector')
        feature_maps = artifacts.get('feature_maps', {})
        config = artifacts.get('config', {})
        
        use_log_transform = config.get('use_log_transform', False)
        use_feature_selection = config.get('use_feature_selection', False)
        prediction_boost_factor = config.get('prediction_boost_factor', 1.1)

        # Enhanced product encoding with fallback
        try:
            if hasattr(product_encoder, 'transform'):
                product_encoded = product_encoder.transform(prediction_input[['nama_produk']])
                product_df = pd.DataFrame(
                    product_encoded,
                    columns=product_encoder.get_feature_names_out(['nama_produk']),
                    index=prediction_input.index
                )
                prediction_row_featured = prediction_input.drop(
                    columns=['nama_produk', 'kategori_produk', 'waktu'], errors='ignore'
                )
                prediction_row_featured = pd.concat([prediction_row_featured, product_df], axis=1)
            else:
                # Fallback: create binary encoding for the product
                prediction_row_featured = prediction_input.drop(
                    columns=['nama_produk', 'kategori_produk', 'waktu'], errors='ignore'
                )
                prediction_row_featured[f'is_product_{product_name}'] = 1
                
        except Exception as e:
            logger.warning(f"Product encoding failed: {e}. Using fallback method.")
            prediction_row_featured = prediction_input.drop(
                columns=['nama_produk', 'kategori_produk', 'waktu'], errors='ignore'
            )

        # Enhanced feature alignment with intelligent defaults
        X_aligned = pd.DataFrame(columns=all_features, index=prediction_row_featured.index)
        
        # Get product-specific defaults
        product_defaults = artifacts.get('feature_maps', {}).get(f"{product_name}_defaults", {})
        global_stats = feature_maps.get('global_stats', {})
        product_popularity = feature_maps.get('product_popularity', {}).get(product_name, 1)
        
        # Extract month for seasonal adjustments
        month = target_date.month if hasattr(target_date, 'month') else prediction_input.get('bulan', 1).iloc[0]
        seasonal_multiplier = feature_maps.get('seasonal_multipliers', {}).get(month, 1)
        
        # Fill features with intelligent defaults
        for col in all_features:
            if col in prediction_row_featured.columns:
                X_aligned[col] = prediction_row_featured[col]
            else:
                # Enhanced fallback logic
                if col in product_defaults:
                    # Use product-specific defaults with preference for recent values
                    if isinstance(product_defaults[col], dict):
                        default_value = product_defaults[col].get('recent', product_defaults[col].get('median', 0))
                    else:
                        default_value = product_defaults[col]
                    X_aligned[col] = default_value
                elif 'rolling_mean' in col or 'penjualan_bulan_lalu' in col:
                    # Use historical patterns enhanced with product popularity
                    base_value = global_stats.get('overall_mean', 10) * product_popularity * seasonal_multiplier
                    X_aligned[col] = base_value
                elif 'seasonal_multiplier' in col:
                    X_aligned[col] = seasonal_multiplier
                elif 'monthly_pattern' in col:
                    monthly_patterns = product_defaults.get('monthly_patterns', {})
                    X_aligned[col] = monthly_patterns.get(month, global_stats.get('overall_mean', 10) * product_popularity)
                elif 'product_popularity' in col:
                    X_aligned[col] = product_popularity
                elif 'category_' in col:
                    category = prediction_input.get('kategori_produk', 'Unknown').iloc[0] if not prediction_input.empty else 'Unknown'
                    category_performance = feature_maps.get('category_performance', {}).get(category, global_stats.get('overall_mean', 10))
                    X_aligned[col] = category_performance
                elif any(x in col for x in ['trend_', 'momentum_', 'volatility_']):
                    # Use conservative positive values for trend features
                    X_aligned[col] = max(0, global_stats.get('overall_mean', 10) * 0.1 * product_popularity)
                elif 'price' in col.lower():
                    # Price-related features
                    unit_price = prediction_input.get('harga_satuan', 0).iloc[0] if not prediction_input.empty else 0
                    X_aligned[col] = unit_price if unit_price > 0 else global_stats.get('overall_mean', 10)
                else:
                    # Generic fallback
                    X_aligned[col] = 0

        # Fill any remaining NaN values
        X_aligned = X_aligned.fillna(0)

        # Enhanced scaling with validation
        cols_to_scale = [col for col in numeric_columns if col in X_aligned.columns]
        if cols_to_scale and scaler is not None:
            try:
                original_values = X_aligned[cols_to_scale].copy()
                X_aligned[cols_to_scale] = scaler.transform(X_aligned[cols_to_scale])
                
                # Validate scaling didn't produce extreme values
                for col in cols_to_scale:
                    if abs(X_aligned[col].iloc[0]) > 10:  # Reasonable threshold
                        logger.warning(f"Extreme scaled value for {col}: {X_aligned[col].iloc[0]}")
                        
            except Exception as e:
                logger.warning(f"Scaling failed: {e}. Using unscaled features.")
                X_aligned[cols_to_scale] = original_values

        # Feature selection if enabled
        if use_feature_selection and selector:
            try:
                X_selected = selector.transform(X_aligned[all_features])
                X_final = pd.DataFrame(X_selected, columns=selected_features, index=X_aligned.index)
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}. Using all aligned features.")
                X_final = X_aligned[selected_features] if all(f in X_aligned.columns for f in selected_features) else X_aligned
        else:
            X_final = X_aligned[selected_features] if all(f in X_aligned.columns for f in selected_features) else X_aligned

        # Make prediction with enhanced post-processing
        try:
            raw_prediction = model.predict(X_final)[0]
            logger.info(f"Raw prediction: {raw_prediction}")
            
            # Apply transformations in reverse order
            if use_log_transform:
                raw_prediction = np.expm1(raw_prediction)
                logger.info(f"After exp transform: {raw_prediction}")
            
            # Apply prediction boost to counter systematic underestimation
            boosted_prediction = raw_prediction * prediction_boost_factor
            logger.info(f"After boost factor ({prediction_boost_factor}): {boosted_prediction}")
            
            # Enhanced post-processing based on historical patterns
            historical_context = get_historical_context(product_name, month, artifacts, prediction_input)
            context_adjusted = apply_historical_context(boosted_prediction, historical_context)
            logger.info(f"After historical context adjustment: {context_adjusted}")
            
            # Final bounds checking
            final_prediction = max(1, round(context_adjusted))  # Ensure minimum of 1 unit
            
            logger.info(f"Final prediction for {product_name}: {final_prediction}")
            return final_prediction
            
        except Exception as e:
            logger.error(f"Prediction calculation failed: {e}")
            # Fallback prediction based on historical patterns
            fallback_prediction = calculate_fallback_prediction(product_name, month, artifacts)
            logger.info(f"Using fallback prediction: {fallback_prediction}")
            return fallback_prediction

    except Exception as e:
        logger.error(f"Enhanced prediction failed for {product_name}: {e}")
        st.error(f"Error during enhanced prediction for {product_name}: {e}")
        return calculate_fallback_prediction(product_name, month, artifacts)

def get_historical_context(product_name, month, artifacts, prediction_input):
    """Extract historical context for better prediction adjustment."""
    try:
        feature_maps = artifacts.get('feature_maps', {})
        
        # Get product-specific information
        product_popularity = feature_maps.get('product_popularity', {}).get(product_name, 1)
        seasonal_multiplier = feature_maps.get('seasonal_multipliers', {}).get(month, 1)
        global_stats = feature_maps.get('global_stats', {})
        
        # Get monthly pattern for this product
        monthly_patterns = feature_maps.get('monthly_patterns', {})
        monthly_pattern = monthly_patterns.get((product_name, month), global_stats.get('overall_mean', 10))
        
        context = {
            'product_popularity': product_popularity,
            'seasonal_multiplier': seasonal_multiplier,
            'monthly_pattern': monthly_pattern,
            'global_mean': global_stats.get('overall_mean', 10),
            'global_std': global_stats.get('overall_std', 5)
        }
        
        return context
        
    except Exception as e:
        logger.warning(f"Failed to get historical context: {e}")
        return {
            'product_popularity': 1,
            'seasonal_multiplier': 1,
            'monthly_pattern': 10,
            'global_mean': 10,
            'global_std': 5
        }

def apply_historical_context(prediction, context):
    """Apply historical context to improve prediction accuracy."""
    try:
        # Base adjustment using multiple factors
        seasonal_weight = 0.3
        popularity_weight = 0.4
        pattern_weight = 0.3
        
        # Calculate context-based expectation
        expected_value = (
            context['monthly_pattern'] * pattern_weight +
            context['global_mean'] * context['product_popularity'] * popularity_weight +
            context['global_mean'] * context['seasonal_multiplier'] * seasonal_weight
        )
        
        # Blend model prediction with context expectation
        # Give more weight to context if prediction is very low compared to expectation
        prediction_ratio = prediction / max(expected_value, 1)
        
        if prediction_ratio < 0.3:  # Model prediction is very low
            blend_weight = 0.7  # Favor historical context more
        elif prediction_ratio < 0.6:  # Moderately low
            blend_weight = 0.5  # Equal weight
        else:  # Reasonable prediction
            blend_weight = 0.3  # Favor model prediction more
            
        adjusted_prediction = (
            prediction * (1 - blend_weight) + 
            expected_value * blend_weight
        )
        
        # Ensure the adjustment is reasonable
        max_adjustment = expected_value * 2
        min_adjustment = max(1, expected_value * 0.1)
        
        final_adjusted = np.clip(adjusted_prediction, min_adjustment, max_adjustment)
        
        return final_adjusted
        
    except Exception as e:
        logger.warning(f"Context adjustment failed: {e}")
        return max(prediction, 1)

def calculate_fallback_prediction(product_name, month, artifacts):
    """Calculate fallback prediction when main prediction fails."""
    try:
        feature_maps = artifacts.get('feature_maps', {})
        global_stats = feature_maps.get('global_stats', {})
        
        # Use simple heuristic based on available information
        product_popularity = feature_maps.get('product_popularity', {}).get(product_name, 1)
        seasonal_multiplier = feature_maps.get('seasonal_multipliers', {}).get(month, 1)
        global_mean = global_stats.get('overall_mean', 10)
        
        fallback = global_mean * product_popularity * seasonal_multiplier
        return max(1, round(fallback))
        
    except Exception as e:
        logger.error(f"Fallback calculation failed: {e}")
        return 5  # Absolute fallback

def generate_enhanced_recommendations(results_df, lang, product_name=None):
    """Enhanced recommendations with better insights."""
    if results_df.empty:
        return ""

    recommendations = []
    
    if product_name:
        pred_col = 'Prediksi Penjualan' if lang == 'ID' else 'Sales Prediction'
        
        # Ensure the prediction column exists
        if pred_col not in results_df.columns:
            # Fallback to the first numeric column if the expected one isn't there
            numeric_cols = results_df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                pred_col = numeric_cols[0]
            else:
                return "" # No numeric data to analyze

        highest_month = results_df.loc[results_df[pred_col].idxmax()]
        lowest_month = results_df.loc[results_df[pred_col].idxmin()]
        average_sales = results_df[pred_col].mean()
        total_predicted = results_df[pred_col].sum()
        
        # --- Language-specific text ---
        if lang == 'ID':
            recommendations.append(f"#### 💡 Rekomendasi Strategis untuk {product_name}")
            recommendations.append(
                f"**📈 Momentum Terkuat:** Siapkan stok ekstra untuk **{highest_month['Periode']}** "
                f"dengan prediksi **{int(highest_month[pred_col])} unit** ({((highest_month[pred_col]/average_sales-1)*100):+.1f}% dari rata-rata). "
                f"Ini adalah peluang terbaik untuk memaksimalkan penjualan!"
            )
            if results_df[pred_col].std() / average_sales > 0.3:
                recommendations.append(
                    f"**⚡ Volatilitas Tinggi:** Penjualan bervariasi signifikan ({(results_df[pred_col].std() / average_sales):.1%}). "
                    f"Fokus pada strategi fleksibel dan responsif terhadap perubahan permintaan."
                )
            if lowest_month['Periode'] != highest_month['Periode']:
                performance_gap = (highest_month[pred_col] - lowest_month[pred_col]) / average_sales * 100
                recommendations.append(
                    f"**📉 Perhatian Khusus:** Bulan **{lowest_month['Periode']}** diprediksi lemah "
                    f"dengan **{int(lowest_month[pred_col])} unit** (gap {performance_gap:.0f}% dari puncak). "
                    f"Pertimbangkan kampanye promosi atau bundling produk."
                )
            # Trend analysis
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
            except:
                trend_desc, action = "**pola yang sulit diprediksi**", "Lakukan analisis mendalam untuk memahami faktor-faktor yang mempengaruhi."
            recommendations.append(
                f"**📊 Analisis Tren:** Selama {len(results_df)} bulan ke depan, {product_name} menunjukkan {trend_desc} "
                f"dengan rata-rata **{average_sales:.1f} unit/bulan** (total: **{total_predicted:.0f} unit**). {action}"
            )
            # Performance benchmarking
            if average_sales >= 50: performance_tier, strategy = "produk unggulan", "Pertahankan posisi dengan inovasi berkelanjutan"
            elif average_sales >= 20: performance_tier, strategy = "produk potensial", "Tingkatkan visibilitas dan jangkauan pasar"
            else: performance_tier, strategy = "produk niche", "Fokus pada segmen spesifik atau evaluasi portfolio"
            recommendations.append(
                f"**🎯 Posisi Strategis:** Dengan prediksi rata-rata {average_sales:.1f} unit/bulan, "
                f"{product_name} termasuk dalam kategori **{performance_tier}**. {strategy}."
            )
        else: # English
            recommendations.append(f"#### 💡 Strategic Recommendations for {product_name}")
            recommendations.append(
                f"**📈 Strongest Momentum:** Prepare extra stock for **{highest_month['Period']}** "
                f"with a prediction of **{int(highest_month[pred_col])} units** ({((highest_month[pred_col]/average_sales-1)*100):+.1f}% of average). "
                f"This is the best opportunity to maximize sales!"
            )
            if results_df[pred_col].std() / average_sales > 0.3:
                recommendations.append(
                    f"**⚡ High Volatility:** Sales vary significantly ({(results_df[pred_col].std() / average_sales):.1%}). "
                    f"Focus on flexible and responsive strategies to demand changes."
                )
            if lowest_month['Period'] != highest_month['Period']:
                performance_gap = (highest_month[pred_col] - lowest_month[pred_col]) / average_sales * 100
                recommendations.append(
                    f"**📉 Special Attention:** The month of **{lowest_month['Period']}** is predicted to be weak "
                    f"with **{int(lowest_month[pred_col])} units** (a {performance_gap:.0f}% gap from the peak). "
                    f"Consider promotional campaigns or product bundling."
                )
            # Trend analysis
            x = np.arange(len(results_df))
            y = results_df[pred_col].values
            try:
                slope, _, r_value, _, _ = stats.linregress(x, y)
                trend_strength = abs(r_value)
                if slope > 0.5 and trend_strength > 0.5: trend_desc, action = "**a strong growth trend** 📈", "Consider capacity expansion and aggressive marketing."
                elif slope > 0.1 and trend_strength > 0.3: trend_desc, action = "**a moderate growth trend** 📊", "Maintain momentum with quality consistency and service."
                elif slope < -0.5 and trend_strength > 0.5: trend_desc, action = "**a concerning downward trend** 📉", "Evaluate product strategy and innovate immediately."
                elif slope < -0.1 and trend_strength > 0.3: trend_desc, action = "**a slight downward trend** ⚠️", "Monitor closely and prepare recovery strategies."
                else: trend_desc, action = "**a relatively stable pattern** ➡️", "Focus on operational efficiency and customer retention."
            except:
                trend_desc, action = "**a hard-to-predict pattern**", "Conduct in-depth analysis to understand influencing factors."
            recommendations.append(
                f"**📊 Trend Analysis:** Over the next {len(results_df)} months, {product_name} shows {trend_desc} "
                f"with an average of **{average_sales:.1f} units/month** (total: **{total_predicted:.0f} units**). {action}"
            )
            # Performance benchmarking
            if average_sales >= 50: performance_tier, strategy = "flagship product", "Maintain position with continuous innovation"
            elif average_sales >= 20: performance_tier, strategy = "potential product", "Increase visibility and market reach"
            else: performance_tier, strategy = "niche product", "Focus on specific segments or evaluate the portfolio"
            recommendations.append(
                f"**🎯 Strategic Position:** With an average prediction of {average_sales:.1f} units/month, "
                f"{product_name} falls into the **{performance_tier}** category. {strategy}."
            )

    else:  # Batch prediction analysis
        pred_col = 'prediksi_penjualan' if lang == 'ID' else 'sales_prediction'
        
        # Enhanced batch analysis
        product_summary = results_df.groupby('nama_produk').agg({
            pred_col: ['sum', 'mean', 'std', 'count']
        }).round(2)
        product_summary.columns = ['Total', 'Average', 'StdDev', 'Periods'] if lang == 'EN' else ['Total', 'Rata-rata', 'StdDev', 'Periode']
        product_summary = product_summary.sort_values('Total', ascending=False)
        
        if lang == 'ID':
            recommendations.append("#### 💡 Analisis Komprehensif Prediksi Batch")
            top_3_products = product_summary.head(3)
            recommendations.append("**⭐ Produk Unggulan (Top 3):**")
            for i, (product, row) in enumerate(top_3_products.iterrows(), 1):
                recommendations.append(
                    f"{i}. **{product}**: {int(row['Total'])} unit total "
                    f"({row['Rata-rata']:.1f}/bulan, variasi: {(row['StdDev']/row['Rata-rata']*100):.0f}%)"
                )
            bottom_products = product_summary.tail(2)
            if len(bottom_products) > 0:
                recommendations.append("**⚠️ Produk Memerlukan Perhatian:**")
                for product, row in bottom_products.iterrows():
                    stability = "stabil" if row['StdDev']/row['Rata-rata'] < 0.5 else "fluktuatif"
                    recommendations.append(
                        f"• **{product}**: Prediksi rendah ({row['Rata-rata']:.1f}/bulan, {stability}). "
                        f"Evaluasi positioning atau bundling dengan produk unggulan."
                    )
            total_revenue_potential = results_df[pred_col].sum()
            avg_monthly_performance = results_df[pred_col].mean()
            portfolio_diversity = len(results_df['nama_produk'].unique())
            recommendations.append(
                f"**💰 Potensi Portfolio:** Total prediksi {int(total_revenue_potential)} unit "
                f"dari {portfolio_diversity} produk (rata-rata: {avg_monthly_performance:.1f} unit/produk/periode). "
            )
            cv_by_product = product_summary['StdDev'] / product_summary['Rata-rata']
            high_risk_products = cv_by_product[cv_by_product > 0.7].index
            if len(high_risk_products) > 0:
                recommendations.append(
                    f"**⚠️ Manajemen Risiko:** {len(high_risk_products)} produk menunjukkan volatilitas tinggi. "
                    f"Pertimbangkan strategi hedging atau diversifikasi lebih lanjut."
                )
        else: # English
            recommendations.append("#### 💡 Comprehensive Batch Prediction Analysis")
            top_3_products = product_summary.head(3)
            recommendations.append("**⭐ Top Products (Top 3):**")
            for i, (product, row) in enumerate(top_3_products.iterrows(), 1):
                recommendations.append(
                    f"{i}. **{product}**: {int(row['Total'])} total units "
                    f"({row['Average']:.1f}/month, variation: {(row['StdDev']/row['Average']*100):.0f}%)"
                )
            bottom_products = product_summary.tail(2)
            if len(bottom_products) > 0:
                recommendations.append("**⚠️ Products Needing Attention:**")
                for product, row in bottom_products.iterrows():
                    stability = "stable" if row['StdDev']/row['Average'] < 0.5 else "volatile"
                    recommendations.append(
                        f"• **{product}**: Low prediction ({row['Average']:.1f}/month, {stability}). "
                        f"Evaluate positioning or bundle with top products."
                    )
            total_revenue_potential = results_df[pred_col].sum()
            avg_monthly_performance = results_df[pred_col].mean()
            portfolio_diversity = len(results_df['nama_produk'].unique())
            recommendations.append(
                f"**💰 Portfolio Potential:** Total prediction of {int(total_revenue_potential)} units "
                f"from {portfolio_diversity} products (average: {avg_monthly_performance:.1f} units/product/period). "
            )
            cv_by_product = product_summary['StdDev'] / product_summary['Average']
            high_risk_products = cv_by_product[cv_by_product > 0.7].index
            if len(high_risk_products) > 0:
                recommendations.append(
                    f"**⚠️ Risk Management:** {len(high_risk_products)} products show high volatility. "
                    f"Consider hedging strategies or further diversification."
                )

    return "\n\n".join(recommendations)

# --- Main Application ---
if lang == 'ID':
    st.markdown("""
        <h1>🔮 Dasbor Prediksi Penjualan</h1>
        <p style="font-size: 0.9rem;">Manfaatkan kekuatan Machine Learning untuk memproyeksikan penjualan dan mengoptimalkan strategi bisnis Anda.</p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <h1>🔮 Sales Prediction Dashboard</h1>
        <p style="font-size: 0.9rem;">Leverage the power of Machine Learning to project sales and optimize your business strategy.</p>
    """, unsafe_allow_html=True)

if mode == 'Advanced':
    st.markdown(f'<span class="advanced-badge">{"Mode Lanjutan" if lang == "ID" else "Advanced Mode"}</span>', unsafe_allow_html=True)

df_raw = st.session_state.get('df', pd.DataFrame())
models, scaler, artifacts = load_artifacts()

if df_raw.empty:
    st.error(f"❌ {('Tidak ada data. Silakan unggah data di halaman Utama.' if lang == 'ID' else 'No data. Please upload data on the Home page.')}")
elif not models or not scaler or not artifacts:
    st.error(f"❌ {('Model atau artefak tidak ditemukan!' if lang == 'ID' else 'Models or artifacts not found!')}")
    st.warning(f"{'> Jalankan skrip `train_model.py` untuk melatih model.' if lang == 'ID' else '> Run the `train_model.py` script to train the models.'}")
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

    model_name_mapping_id = {
        'Gabungan': 'Gabungan Model Terbaik (Disarankan)',
        'XGBoost': 'Akurasi Tinggi & Cepat',
        'LightGBM': 'Efisien & Stabil', 
        'Random Forest': 'Robust & Andal',
        'Gradient Boosting': 'Presisi Tinggi'
    }
    model_name_mapping_en = {
        'Gabungan': 'Best Ensemble Model (Recommended)',
        'XGBoost': 'High Accuracy & Fast',
        'LightGBM': 'Efficient & Stable',
        'Random Forest': 'Robust & Reliable',
        'Gradient Boosting': 'High Precision'
    }
    model_name_mapping = model_name_mapping_id if lang == 'ID' else model_name_mapping_en
    reversed_model_mapping = {v: k for k, v in model_name_mapping.items()}

    # --- UI Tabs ---
    tab1_text = "Prediksi 1 Produk" if lang == 'ID' else "Predict 1 Product"
    tab2_text = "Prediksi Banyak Produk" if lang == 'ID' else "Predict Multiple Products"
    if mode == 'Advanced':
        tab1_text = f"🎯 {tab1_text}"
        tab2_text = f"📦 {tab2_text}"

    tab1, tab2 = st.tabs([tab1_text, tab2_text])

    # --- TAB 1: Interactive Prediction ---
    with tab1:
        if lang == 'ID':
            st.header("🎯 Prediksi Penjualan Interaktif")
            st.markdown("Dapatkan prediksi akurat untuk satu produk pilihan Anda guna merencanakan strategi ke depan.")
        else:
            st.header("🎯 Interactive Sales Prediction")
            st.markdown("Get accurate predictions for a single product of your choice to plan your future strategy.")
        
        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                product_list = sorted(df_raw['nama_produk'].unique())
                selected_product = st.selectbox(
                    "Pilih Produk" if lang == 'ID' else "Select Product", 
                    product_list, 
                    key='interactive_product',
                    help="Pilih produk untuk dianalisis." if lang == 'ID' else "Select a product to analyze."
                )
            with col2:
                prediction_months = st.slider(
                    "Horizon Prediksi (bulan)" if lang == 'ID' else "Prediction Horizon (months)", 
                    1, 18, 6, 
                    key='interactive_months',
                    help="Semakin jauh horizon, semakin tinggi ketidakpastian prediksi." if lang == 'ID' else "The further the horizon, the higher the prediction uncertainty."
                )

            if mode == 'Advanced':
                col3, col4 = st.columns(2)
                with col3:
                    model_list = [k for k in models.keys() if k != 'Gabungan'] + ['Gabungan']
                    selected_model_name_advanced = st.selectbox(
                        "Pilih Model" if lang == 'ID' else "Select Model", 
                        model_list, 
                        index=len(model_list)-1, 
                        key='interactive_model_adv'
                    )
                with col4:
                    show_confidence = st.toggle("Tampilkan Interval Kepercayaan" if lang == 'ID' else "Show Confidence Interval", value=True)
            else: # Normal Mode
                normal_model_options = list(model_name_mapping.values())
                selected_model_name_normal = st.selectbox(
                    "Pilih Pendekatan" if lang == 'ID' else "Select Approach", 
                    normal_model_options, 
                    index=0, 
                    key='interactive_model_normal'
                )
                show_confidence = False

        if st.button("Buat Prediksi" if lang == 'ID' else "Generate Prediction", type="primary", key='interactive_button', use_container_width=True):
            selected_model_name = selected_model_name_advanced if mode == 'Advanced' else reversed_model_mapping[selected_model_name_normal]

            spinner_text = f"Menganalisis pola penjualan {selected_product}..." if lang == 'ID' else f"Analyzing sales patterns for {selected_product}..."
            with st.spinner(f"🧠 {spinner_text}"):
                try:
                    feature_maps = artifacts['feature_maps']
                    product_history = df_agg[df_agg['nama_produk'] == selected_product].copy()
                    prediction_results = []
                    last_known_date = product_history['waktu'].max() if not product_history.empty else df_raw['waktu'].max()

                    for i in range(1, prediction_months + 1):
                        prediction_date = last_known_date + relativedelta(months=i)
                        pred_input_row = pd.DataFrame([{
                            'waktu': prediction_date, 'nama_produk': selected_product, 'jumlah': 0, 'harga': 0, 
                            'harga_satuan': latest_prices.get(selected_product, 0), 
                            'kategori_produk': category_map.get(selected_product, 'N/A'), 
                            'tahun': prediction_date.year, 'bulan': prediction_date.month
                        }])
                        
                        combined_df = pd.concat([product_history, pred_input_row], ignore_index=True)
                        enhanced_df, _ = create_advanced_features(combined_df, feature_maps, is_training=False)
                        prediction_row_featured = enhanced_df.iloc[-1:].copy()
                        
                        model = models[selected_model_name]
                        predicted_value = enhanced_predict_sales(
                            model, prediction_row_featured, artifacts, scaler, 
                            selected_product, prediction_date
                        )
                        
                        prediction_results.append({
                            ('Periode' if lang == 'ID' else 'Period'): prediction_date.strftime("%Y-%m"), 
                            ('Prediksi Penjualan' if lang == 'ID' else 'Sales Prediction'): predicted_value
                        })
                        
                        new_history_row = pred_input_row.copy()
                        new_history_row['jumlah'] = predicted_value
                        product_history = pd.concat([product_history, new_history_row], ignore_index=True)

                    st.success("✅ " + ("Prediksi berhasil dibuat!" if lang == 'ID' else "Prediction generated successfully!"))
                    results_df = pd.DataFrame(prediction_results)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.subheader("📊 " + ("Hasil Prediksi" if lang == 'ID' else "Prediction Results"))
                    with col2:
                        total_predicted = results_df.iloc[:, 1].sum()
                        st.metric("Total " + ("Prediksi" if lang == 'ID' else "Prediction"), f"{total_predicted:,.0f} " + ("unit" if lang == 'ID' else "units"))
                    
                    st.dataframe(results_df, use_container_width=True)

                    st.subheader("📈 " + ("Visualisasi Tren" if lang == 'ID' else "Trend Visualization"))
                    fig = go.Figure()
                    
                    hist_prod_df = df_agg[df_agg['nama_produk'] == selected_product]
                    if not hist_prod_df.empty:
                        fig.add_trace(go.Scatter(
                            x=hist_prod_df['waktu'], y=hist_prod_df['jumlah'], mode='lines+markers',
                            name='📈 ' + ('Penjualan Historis' if lang == 'ID' else 'Historical Sales'),
                            line=dict(color='#2E86C1', width=3), marker=dict(size=6)
                        ))
                    
                    pred_dates = pd.to_datetime(results_df.iloc[:, 0])
                    pred_values = results_df.iloc[:, 1]
                    
                    if mode == 'Advanced' and show_confidence:
                        uncertainty = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35] * 3)[:len(pred_values)]
                        upper_bound = pred_values * (1 + uncertainty)
                        lower_bound = pred_values * (1 - uncertainty)
                        fig.add_trace(go.Scatter(x=pred_dates, y=upper_bound, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                        fig.add_trace(go.Scatter(
                            x=pred_dates, y=lower_bound, fill='tonexty', mode='lines', 
                            line_color='rgba(255,0,0,0)', name='🎯 ' + ('Interval Kepercayaan' if lang == 'ID' else 'Confidence Interval'),
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=pred_dates, y=pred_values, mode='lines+markers',
                        name='🔮 ' + ('Prediksi' if lang == 'ID' else 'Prediction'),
                        line=dict(color='#E74C3C', width=3, dash='dot'), marker=dict(size=8, symbol='diamond')
                    ))
                    
                    fig.update_layout(
                        title=f"{('Prediksi untuk' if lang == 'ID' else 'Prediction for')} {selected_product} | Model: {selected_model_name}",
                        xaxis_title='Periode' if lang == 'ID' else 'Period',
                        yaxis_title='Jumlah Penjualan (Unit)' if lang == 'ID' else 'Sales Volume (Units)',
                        template=plotly_template, height=500, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    recommendation_text = generate_enhanced_recommendations(results_df, lang, selected_product)
                    st.markdown(recommendation_text)

                except Exception as e:
                    st.error(f"❌ {('Gagal membuat prediksi:' if lang == 'ID' else 'Failed to generate prediction:')} {e}")
                    logger.error(f"Interactive prediction failed: {e}", exc_info=True)

    # --- TAB 2: Batch Prediction ---
    with tab2:
        if lang == 'ID':
            st.header("📦 Prediksi Batch untuk Banyak Produk")
            st.markdown("Analisis prediksi untuk beberapa produk secara bersamaan untuk mendapatkan wawasan portofolio.")
        else:
            st.header("📦 Batch Prediction for Multiple Products")
            st.markdown("Analyze predictions for multiple products simultaneously to gain portfolio insights.")

        with st.container(border=True):
            all_products = sorted(df_raw['nama_produk'].unique())
            
            col1, col2 = st.columns(2)
            with col1:
                selection_method = st.radio(
                    "Metode Pemilihan Produk:" if lang == 'ID' else "Product Selection Method:",
                    ["Top Performers", "Pilih Manual", "Semua Produk"] if lang == 'ID' else ["Top Performers", "Manual Select", "All Products"],
                    help="Pilih cara untuk menentukan produk yang akan dianalisis" if lang == 'ID' else "Choose how to select products for analysis"
                )
            
            with col2:
                if selection_method == ("Top Performers"):
                    top_count = st.slider("Jumlah produk top" if lang == 'ID' else "Number of top products", 3, min(20, len(all_products)), 5)
                    product_sales = df_raw.groupby('nama_produk')['jumlah'].sum().sort_values(ascending=False)
                    selected_products_batch = product_sales.head(top_count).index.tolist()
                    st.info(f"✅ {('Dipilih' if lang == 'ID' else 'Selected')} {len(selected_products_batch)} {('produk dengan penjualan tertinggi' if lang == 'ID' else 'top-selling products')}")
                elif selection_method == ("Pilih Manual" if lang == 'ID' else "Manual Select"):
                    selected_products_batch = st.multiselect(
                        "Pilih Produk" if lang == 'ID' else "Select Products", 
                        all_products, default=all_products[:5], key='batch_products'
                    )
                else: # All Products
                    selected_products_batch = all_products
                    st.info(f"✅ {('Menganalisis semua' if lang == 'ID' else 'Analyzing all')} {len(all_products)} {('produk' if lang == 'ID' else 'products')}")
            
            col3, col4 = st.columns(2)
            with col3:
                prediction_start_date = st.date_input(
                    "Tanggal Mulai Prediksi" if lang == 'ID' else "Prediction Start Date", 
                    datetime.now(), help="Pilih tanggal mulai untuk periode prediksi" if lang == 'ID' else "Select the start date for the prediction period"
                )
            with col4:
                months_to_predict_batch = st.number_input(
                    "Periode Prediksi (bulan)" if lang == 'ID' else "Prediction Period (months)", 
                    min_value=1, max_value=24, value=6, key='batch_months'
                )

            if mode == 'Advanced':
                col5, col6 = st.columns(2)
                with col5:
                    model_list_batch = [k for k in models.keys() if k != 'Gabungan'] + ['Gabungan']
                    selected_model_batch_adv = st.selectbox(
                        "Model" if lang == 'ID' else "Model", 
                        model_list_batch, index=len(model_list_batch)-1, key='batch_model_adv'
                    )
                with col6:
                    include_analytics = st.toggle("Analisis Portofolio" if lang == 'ID' else "Portfolio Analysis", value=True)
            else: # Normal Mode
                normal_model_options_batch = list(model_name_mapping.values())
                selected_model_batch_normal = st.selectbox(
                    "Pendekatan" if lang == 'ID' else "Approach", 
                    normal_model_options_batch, index=0, key='batch_model_normal'
                )
                include_analytics = True

        if st.button("Jalankan Analisis Batch" if lang == 'ID' else "Run Batch Analysis", type="primary", key='batch_button', use_container_width=True):
            if not selected_products_batch:
                st.warning("⚠️ " + ("Silakan pilih setidaknya satu produk untuk dianalisis." if lang == 'ID' else "Please select at least one product to analyze."))
            else:
                selected_model_batch = selected_model_batch_adv if mode == 'Advanced' else reversed_model_mapping[selected_model_batch_normal]

                progress_bar = st.progress(0)
                status_text = st.empty()
                total_operations = len(selected_products_batch) * months_to_predict_batch
                current_operation = 0

                spinner_text_batch = f"Menganalisis {len(selected_products_batch)} produk..." if lang == 'ID' else f"Analyzing {len(selected_products_batch)} products..."
                with st.spinner(f"🧠 {spinner_text_batch}"):
                    try:
                        feature_maps = artifacts['feature_maps']
                        batch_results = []
                        
                        for product_idx, product in enumerate(selected_products_batch):
                            status_text.text(f"{('Memproses' if lang == 'ID' else 'Processing')}: {product} ({product_idx + 1}/{len(selected_products_batch)})")
                            
                            product_history = df_agg[df_agg['nama_produk'] == product].copy()
                            
                            for month_idx in range(months_to_predict_batch):
                                prediction_date = prediction_start_date + relativedelta(months=month_idx)
                                pred_input_row = pd.DataFrame([{'waktu': prediction_date, 'nama_produk': product, 'jumlah': 0, 'harga': 0, 'harga_satuan': latest_prices.get(product, 0), 'kategori_produk': category_map.get(product, 'N/A'), 'tahun': prediction_date.year, 'bulan': prediction_date.month}])
                                combined_df = pd.concat([product_history, pred_input_row], ignore_index=True)
                                enhanced_df, _ = create_advanced_features(combined_df, feature_maps, is_training=False)
                                prediction_row_featured = enhanced_df.iloc[-1:].copy()
                                model = models[selected_model_batch]
                                predicted_value = enhanced_predict_sales(model, prediction_row_featured, artifacts, scaler, product, prediction_date)
                                
                                batch_results.append({
                                    'nama_produk': product, 
                                    'tahun': prediction_date.year, 
                                    'bulan': prediction_date.strftime('%Y-%m'), 
                                    ('prediksi_penjualan' if lang == 'ID' else 'sales_prediction'): predicted_value,
                                    'kategori': category_map.get(product, 'Unknown'),
                                    'harga_satuan': latest_prices.get(product, 0)
                                })
                                current_operation += 1
                                progress_bar.progress(current_operation / total_operations)

                        progress_bar.progress(1.0)
                        status_text.text("✅ " + ("Analisis selesai!" if lang == 'ID' else "Analysis complete!"))
                        
                        st.success(f"✅ {('Analisis portofolio berhasil untuk' if lang == 'ID' else 'Portfolio analysis successful for')} {len(selected_products_batch)} {('produk!' if lang == 'ID' else 'products!')}")
                        results_df = pd.DataFrame(batch_results)
                        
                        st.subheader("📊 " + ("Dasbor Hasil Analisis" if lang == 'ID' else "Analysis Results Dashboard"))
                        
                        col1, col2, col3, col4 = st.columns(4)
                        pred_col_name = 'prediksi_penjualan' if lang == 'ID' else 'sales_prediction'
                        with col1:
                            total_units = results_df[pred_col_name].sum()
                            st.metric("Total " + ("Prediksi" if lang == 'ID' else "Prediction"), f"{total_units:,.0f}", "unit" if lang == 'ID' else "units")
                        with col2:
                            avg_per_product = results_df.groupby('nama_produk')[pred_col_name].sum().mean()
                            st.metric("Rata-rata/Produk" if lang == 'ID' else "Avg/Product", f"{avg_per_product:,.0f}", "unit" if lang == 'ID' else "units")
                        with col3:
                            best_month = results_df.groupby('bulan')[pred_col_name].sum().idxmax()
                            st.metric("Bulan Terbaik" if lang == 'ID' else "Best Month", best_month)
                        with col4:
                            total_revenue_est = (results_df[pred_col_name] * results_df['harga_satuan']).sum()
                            st.metric("Est. Pendapatan" if lang == 'ID' else "Est. Revenue", f"Rp {total_revenue_est:,.0f}")
                        
                        st.subheader("📋 " + ("Detail Hasil Prediksi" if lang == 'ID' else "Detailed Prediction Results"))
                        
                        results_display = results_df.copy()
                        results_display['estimasi_revenue'] = results_display[pred_col_name] * results_display['harga_satuan']
                        results_display = results_display.round(2)
                        st.dataframe(results_display, use_container_width=True)
                        
                        if include_analytics:
                            st.markdown("---")
                            st.subheader("📊 " + ("Analisis Portofolio Mendalam" if lang == 'ID' else "In-depth Portfolio Analysis"))
                            
                            pivot_data = results_df.pivot_table(values=pred_col_name, index='nama_produk', columns='bulan', aggfunc='sum').fillna(0)
                            
                            category_performance = results_df.groupby('kategori').agg({
                                pred_col_name: ['sum', 'mean', 'count'],
                                'harga_satuan': 'mean'
                            }).round(2)
                            category_performance.columns = ["Total_Unit", "Rata_Unit", "Jumlah_Produk", "Harga_Rata"] if lang == 'ID' else ["Total_Units", "Avg_Units", "Num_Products", "Avg_Price"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**" + ("Performa per Kategori:" if lang == 'ID' else "Performance by Category:") + "**")
                                st.dataframe(category_performance, use_container_width=True)
                            with col2:
                                top_products = results_df.groupby('nama_produk')[pred_col_name].sum().sort_values(ascending=True).tail(10)
                                fig_bar = go.Figure(go.Bar(x=top_products.values, y=top_products.index, orientation='h', marker_color='#3498DB'))
                                fig_bar.update_layout(
                                    title="🏆 " + ("Top 10 Produk (Prediksi Total)" if lang == 'ID' else "Top 10 Products (Total Prediction)"),
                                    xaxis_title="Unit" if lang == 'ID' else "Units", height=400, template=plotly_template
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)

                        st.markdown("---")
                        recommendation_text = generate_enhanced_recommendations(results_df, lang)
                        st.markdown(recommendation_text)

                        col1, col2 = st.columns(2)
                        with col1:
                            csv_results = results_display.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 " + ("Unduh Hasil Lengkap (.csv)" if lang == 'ID' else "Download Full Results (.csv)"), 
                                csv_results, f'prediksi_portfolio_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', 'text/csv', 
                                key='download_batch_detailed', use_container_width=True
                            )
                        with col2:
                            summary_df = results_df.groupby('nama_produk').agg({
                                pred_col_name: 'sum', 'kategori': 'first'
                            }).round(0).reset_index()
                            summary_df.columns = ['Produk', 'Total_Prediksi', 'Kategori'] if lang == 'ID' else ['Product', 'Total_Prediction', 'Category']
                            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📋 " + ("Unduh Ringkasan (.csv)" if lang == 'ID' else "Download Summary (.csv)"),
                                summary_csv, f'ringkasan_prediksi_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', 'text/csv',
                                key='download_summary', use_container_width=True
                            )

                    except Exception as e:
                        progress_bar.progress(0)
                        status_text.text("")
                        st.error(f"❌ {('Gagal melakukan analisis batch:' if lang == 'ID' else 'Failed to run batch analysis:')} {e}")
                        logger.error(f"Batch prediction failed: {e}", exc_info=True)

    if mode == 'Advanced':
        expander_title = "🔬 Analisis Performa & Konfigurasi Model" if lang == 'ID' else "🔬 Model Performance & Configuration Analysis"
        with st.expander(expander_title):
            try:
                training_results = artifacts['training_results']
                config = artifacts['config']
                timestamp = artifacts['training_timestamp']
                
                st.info(f"📅 **Training Session:** {datetime.fromisoformat(timestamp).strftime('%d %B %Y, %H:%M WIB')}")
                
                perf_df = pd.DataFrame(training_results).T
                perf_df['Grade'] = perf_df['test_r2'].apply(lambda x: 
                    '🏆 Excellent' if x >= 0.9 else '🥇 Very Good' if x >= 0.8 else
                    '🥈 Good' if x >= 0.7 else '🥉 Fair' if x >= 0.6 else '⚠️ Needs Improvement'
                )
                
                st.dataframe(perf_df.style.format({
                    'test_r2': '{:.3f}', 'test_mape': '{:.1f}%', 'test_mae': '{:.2f}',
                    'mean_prediction_test': '{:.1f}', 'mean_actual_test': '{:.1f}'
                }), use_container_width=True)
                
                st.markdown("**⚙️ " + ("Konfigurasi Training:" if lang == 'ID' else "Training Configuration:") + "**")
                config_display = {
                    ("Transformasi Target" if lang == 'ID' else "Target Transform"): "Log Transform" if config.get('use_log_transform') else "Original Scale",
                    ("Seleksi Fitur" if lang == 'ID' else "Feature Selection"): f"Top {config.get('k_best_features', 'N/A')} features" if config.get('use_feature_selection') else "All features",
                    "Boost Factor": f"{config.get('prediction_boost_factor', 1.0):.2f}x",
                    "CV Folds": config.get('cv_folds', 'N/A'),
                    ("Tuning Hyperparameter" if lang == 'ID' else "Hyperparameter Tuning"): "Enabled" if config.get('enable_hyperparameter_tuning') else "Disabled"
                }
                for key, value in config_display.items():
                    st.write(f"**{key}:** {value}")
                
                if 'feature_importance' in artifacts and artifacts['feature_importance']:
                    st.markdown("**🎯 " + ("Pentingnya Fitur (Top 10):" if lang == 'ID' else "Feature Importance (Top 10):") + "**")
                    best_model_name = max(training_results.keys(), key=lambda x: training_results[x]['test_r2'])
                    
                    if best_model_name in artifacts['feature_importance']:
                        importance = artifacts['feature_importance'][best_model_name]
                        importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False).head(10)
                        
                        fig_importance = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'], orientation='h', marker_color='#E74C3C'))
                        fig_importance.update_layout(
                            title=f"🎯 {('Pentingnya Fitur' if lang == 'ID' else 'Feature Importance')} - {best_model_name}",
                            height=400, template=plotly_template
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                
            except KeyError as e:
                st.warning(f"⚠️ {('Informasi performa tidak lengkap:' if lang == 'ID' else 'Incomplete performance information:')} {e}. {('Latih ulang model untuk data lengkap.' if lang == 'ID' else 'Retrain the model for complete data.')}")
            except Exception as e:
                st.error(f"❌ {('Gagal memuat informasi performa:' if lang == 'ID' else 'Failed to load performance information:')} {e}")