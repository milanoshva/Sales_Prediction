
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from dateutil.relativedelta import relativedelta
import warnings
from datetime import datetime
import plotly.graph_objects as go
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import sys
from ui.styles import set_custom_ui, get_plotly_template
from core.data_processor import create_advanced_features, create_prediction_input, optimize_hyperparameters

# --- Setup ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, filename='app.txt', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
set_custom_ui()
plt.style.use('dark_background')
sns.set_palette("husl")
plotly_template = get_plotly_template()

# --- Constants ---
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATHS = {
    'Random Forest': os.path.join(MODEL_DIR, 'model_rf_jumlah.pkl'),
    'XGBoost': os.path.join(MODEL_DIR, 'model_xgb_jumlah.pkl'),
    'LightGBM': os.path.join(MODEL_DIR, 'model_lgb_jumlah.pkl'),
    'Gabungan': os.path.join(MODEL_DIR, 'model_ensemble_jumlah.pkl')
}
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
FEATURE_MAP_PATH = os.path.join(MODEL_DIR, 'feature_maps.pkl')

# Initialize session state for language
if 'language' not in st.session_state:
    st.session_state['language'] = 'ID'

# Global variables
lang = st.session_state.get('language', 'ID')
mode = st.session_state.get('mode', 'Normal') # Initialize mode here

st.markdown(f"""<h1>📈 {'Prediksi Penjualan untuk UMKM Kuliner' if lang == 'ID' else 'Sales Prediction for Culinary SME'}</h1>
<p style="font-size: 0.9rem;">{'Gunakan model machine learning untuk memprediksi penjualan dan pendapatan di masa depan.' if lang == 'ID' else 'Use machine learning models to predict future sales and revenue.'}</p>""", unsafe_allow_html=True)

if 'df' not in st.session_state or st.session_state.df.empty:
    st.error(f"❌ {'Tidak ada data penjualan! Silakan unggah data di halaman Utama terlebih dahulu.' if lang == 'ID' else 'No sales data available! Please upload data on the Home page first.'}")
else:
    df = st.session_state.df.copy()
    if mode == 'Advanced': st.markdown(f'<span class="advanced-badge">{("Mode Lanjutan" if lang == 'ID' else "Advanced Mode")}</span>', unsafe_allow_html=True)

    # --- Initial Data Prep ---
    if not pd.api.types.is_datetime64_any_dtype(df['waktu']):
        df['waktu'] = pd.to_datetime(df['waktu'], errors='coerce')
        df.dropna(subset=['waktu'], inplace=True)
    
    df['tahun'] = df['waktu'].dt.year
    df['bulan'] = df['waktu'].dt.month
    df_sorted = df.sort_values(by='waktu', ascending=False)
    harga_satuan_per_produk = df_sorted.drop_duplicates(subset=['nama_produk'])[['nama_produk', 'harga_satuan']].set_index('nama_produk')['harga_satuan'].to_dict()
    df_agg = df.groupby(['tahun', 'bulan', 'nama_produk']).agg(jumlah=('jumlah', 'sum'), harga=('harga', 'sum')).reset_index()
    kategori_mapping = df[['nama_produk', 'kategori_produk']].drop_duplicates().set_index('nama_produk')['kategori_produk']
    df_agg['kategori_produk'] = df_agg['nama_produk'].map(kategori_mapping)
    df_agg['harga_satuan'] = df_agg['nama_produk'].map(harga_satuan_per_produk)
    df_agg['harga_satuan'].fillna(df_agg['harga_satuan'].mean(), inplace=True)

    # --- Enhanced UI Elements ---
    with st.expander(f"⚙️ {'Pengaturan Model Lanjutan' if lang == 'ID' else 'Advanced Model Settings'}"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_choice = st.selectbox(
                f"{'Pilih Metode Prediksi' if lang == 'ID' else 'Select Prediction Method'}",
                ['Random Forest', 'XGBoost', 'LightGBM', 'Gabungan']
            )
            
            enable_hyperparameter_tuning = st.checkbox(
                f"{'Aktifkan Optimasi Hyperparameter' if lang == 'ID' else 'Enable Hyperparameter Optimization'}", 
                value=True,
                help="Otomatis mencari parameter terbaik untuk model" if lang == 'ID' else "Automatically find best parameters for the model"
            )
            
            handle_outliers = st.checkbox(
                f"{'Hapus Penjualan Ekstrem' if lang == 'ID' else 'Remove Extreme Sales'}", 
                value=True
            )
        
        with col2:
            feature_selection = st.checkbox(
                f"{'Seleksi Fitur Otomatis' if lang == 'ID' else 'Automatic Feature Selection'}", 
                value=True,
                help="Pilih fitur terbaik secara otomatis" if lang == 'ID' else "Automatically select the best features"
            )
            
            use_log_transform = st.checkbox(
                f"{'Gunakan Transformasi Log' if lang == 'ID' else 'Use Log Transformation'}", 
                value=True,
                help="Mengurangi skewness pada data penjualan" if lang == 'ID' else "Reduce skewness in sales data"
            )
            
            cross_validation_folds = st.selectbox(
                f"{'Jumlah Fold untuk Cross Validation' if lang == 'ID' else 'Cross Validation Folds'}",
                [3, 5, 7],
                index=0
            )
        
        if handle_outliers:
            outlier_threshold = st.slider(
                f"{'Batas Penjualan Ekstrem (Std. Dev.)' if lang == 'ID' else 'Extreme Sales Threshold (Std. Dev.)'}", 
                2.0, 5.0, 3.0
            )
        
        if not enable_hyperparameter_tuning and mode == 'Advanced':
            st.write("Manual Parameter Settings:")
            col3, col4 = st.columns(2)
            with col3:
                n_estimators = st.slider(f"{'Jumlah Pohon (n_estimators)' if lang == 'ID' else 'Number of Trees (n_estimators)'}", 100, 1000, 300)
            with col4:
                max_depth = st.slider(f"{'Kedalaman Maksimum (max_depth)' if lang == 'ID' else 'Maximum Depth (max_depth)'}", 5, 30, 15)
        else:
            n_estimators, max_depth = 300, 15

    st.subheader(f"🛠️ {'Latih Model Prediksi' if lang == 'ID' else 'Train Prediction Model'}")
    
    if df_agg.empty or len(df_agg) < 30:
        st.warning(f"{'Tidak cukup data untuk melatih model. Minimal 30 data point diperlukan.' if lang == 'ID' else 'Not enough data to train the model. Minimum 30 data points required.'}")
    else:
        if st.button(f"{'🚀 Latih Model' if lang == 'ID' else '🚀 Train Model'}"):
            with st.spinner(f"{'Mempersiapkan model...' if lang == 'ID' else 'Preparing model...'}"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Data Preparation
                status_text.text("Step 1/6: Data Preparation...")
                progress_bar.progress(10)
                
                df_train_ready = df_agg.sort_values(['tahun', 'bulan']).copy()
                
                # Step 2: Feature Engineering
                status_text.text("Step 2/6: Advanced Feature Engineering...")
                progress_bar.progress(25)
                
                df_enhanced, feature_maps = create_advanced_features(df_train_ready, is_training=True)
                
                # Handle categorical encoding
                df_enhanced['kategori_produk'] = df_enhanced['kategori_produk'].astype('category')
                feature_maps['all_categories'] = df_enhanced['kategori_produk'].cat.categories
                df_enhanced['kategori_encoded'] = df_enhanced['kategori_produk'].cat.codes
                
                # Step 3: Train/Test Split
                status_text.text("Step 3/6: Train/Test Split...")
                progress_bar.progress(40)
                
                tscv = TimeSeriesSplit(n_splits=cross_validation_folds)
                train_idx, test_idx = list(tscv.split(df_enhanced))[-1]
                
                X_train = df_enhanced.iloc[train_idx].copy()
                X_test = df_enhanced.iloc[test_idx].copy()
                y_train = X_train['jumlah'].copy()
                y_test = X_test['jumlah'].copy()
                
                # Remove target from features
                feature_columns = [col for col in X_train.columns if col not in ['jumlah', 'nama_produk', 'kategori_produk']]
                X_train_features = X_train[feature_columns].copy()
                X_test_features = X_test[feature_columns].copy()
                
                # Step 4: Outlier Handling and Scaling
                status_text.text("Step 4/6: Outlier Handling and Scaling...")
                progress_bar.progress(55)

                if handle_outliers:
                    z_scores = np.abs(stats.zscore(y_train))
                    outlier_mask = z_scores < outlier_threshold
                    X_train_features = X_train_features[outlier_mask]
                    y_train = y_train[outlier_mask]

                # Log transformation
                if use_log_transform:
                    y_train_transformed = np.log1p(y_train)
                    y_test_transformed = np.log1p(y_test)
                else:
                    y_train_transformed = y_train
                    y_test_transformed = y_test

                # Scaling
                scaler = RobustScaler()
                numeric_columns = X_train_features.select_dtypes(include=[np.number]).columns

                X_train_scaled = X_train_features.copy()
                X_test_scaled = X_test_features.copy()

                X_train_scaled[numeric_columns] = scaler.fit_transform(X_train_features[numeric_columns])
                X_test_scaled[numeric_columns] = scaler.transform(X_test_features[numeric_columns])

                # Handle NaN after scaling
                X_train_scaled = X_train_scaled.fillna(X_train_scaled.median())
                X_test_scaled = X_test_scaled.fillna(X_train_scaled.median())  # Use train median for consistency

                # Step 5: Feature Selection
                status_text.text("Step 5/6: Feature Selection...")
                progress_bar.progress(70)

                if feature_selection:
                    try:
                        k_best = min(20, len(feature_columns))
                        selector = SelectKBest(score_func=f_regression, k=k_best)

                        # Validate that X_train_scaled does not contain NaN
                        if X_train_scaled.isnull().values.any():
                            raise ValueError("X_train_scaled contains NaN even after fillna.")

                        X_train_selected = selector.fit_transform(X_train_scaled, y_train_transformed)
                        X_test_selected = selector.transform(X_test_scaled)

                        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]

                    except ValueError as e:
                        st.warning(f"⚠️ Feature selection skipped due to: {e}")
                        logger.warning(f"Feature selection failed: {e}")
                        X_train_selected = X_train_scaled
                        X_test_selected = X_test_scaled
                        selected_features = feature_columns
                else:
                    X_train_selected = X_train_scaled
                    X_test_selected = X_test_scaled
                    selected_features = feature_columns

                # Step 6: Model Training and Optimization
                status_text.text("Step 6/6: Model Training and Optimization...")
                progress_bar.progress(85)

                if model_choice == 'Gabungan':
                    if enable_hyperparameter_tuning:
                        rf_optimized = optimize_hyperparameters(X_train_selected, y_train_transformed, 'Random Forest', cross_validation_folds)
                        xgb_optimized = optimize_hyperparameters(X_train_selected, y_train_transformed, 'XGBoost', cross_validation_folds)
                        lgb_optimized = optimize_hyperparameters(X_train_selected, y_train_transformed, 'LightGBM', cross_validation_folds)
                    else:
                        rf_optimized = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                        xgb_optimized = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                        lgb_optimized = LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

                    ensemble_model = VotingRegressor([
                        ('rf', rf_optimized),
                        ('xgb', xgb_optimized),
                        ('lgb', lgb_optimized)
                    ], weights=[0.4, 0.3, 0.3])

                    ensemble_model.fit(X_train_selected, y_train_transformed)
                    trained_model = ensemble_model

                else:
                    if enable_hyperparameter_tuning:
                        trained_model = optimize_hyperparameters(X_train_selected, y_train_transformed, model_choice, cross_validation_folds)
                    else:
                        base_models = {
                            'Random Forest': RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1),
                            'XGBoost': XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1),
                            'LightGBM': LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
                        }
                        trained_model = base_models[model_choice]

                    trained_model.fit(X_train_selected, y_train_transformed)

                progress_bar.progress(100)
                status_text.text("✅ Model training completed!")
                
                # Model Evaluation
                y_pred_train = trained_model.predict(X_train_selected)
                y_pred_test = trained_model.predict(X_test_selected)
                
                # Inverse transform if log was used
                if use_log_transform:
                    y_pred_train = np.expm1(y_pred_train)
                    y_pred_test = np.expm1(y_pred_test)
                    y_train_eval = np.expm1(y_train_transformed)
                    y_test_eval = np.expm1(y_test_transformed)
                else:
                    y_train_eval = y_train_transformed
                    y_test_eval = y_test_transformed
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train_eval, y_pred_train)
                test_mae = mean_absolute_error(y_test_eval, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train_eval, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_test))
                train_r2 = r2_score(y_train_eval, y_pred_train)
                test_r2 = r2_score(y_test_eval, y_pred_test)
                
                # Calculate MAPE
                def mean_absolute_percentage_error(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                
                train_mape = mean_absolute_percentage_error(y_train_eval, y_pred_train)
                test_mape = mean_absolute_percentage_error(y_test_eval, y_pred_test)
                
                progress_bar.progress(100)
                status_text.text("Training completed!")

                # Save to session
                st.session_state[f"{model_choice}_performance"] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mape': train_mape,
                    'test_mape': test_mape
                }
                
                # Save models and artifacts
                joblib.dump(trained_model, MODEL_PATHS[model_choice])
                joblib.dump(scaler, SCALER_PATH)
                joblib.dump({
                    'feature_maps': feature_maps,
                    'all_features': feature_columns,
                    'selected_features': selected_features,
                    'numeric_columns': numeric_columns.tolist(),
                    'use_log_transform': use_log_transform,
                    'feature_selection': feature_selection,
                    'selector': selector if feature_selection else None,
                    'model_choice': model_choice
                }, FEATURE_MAP_PATH)
                
                # Display Results
                notif = "✅ Model berhasil dilatih!" if lang == "ID" else "✅ Model trained successfully!"
                if handle_outliers:
                    notif += f" | {'Dihapus' if lang == 'ID' else 'Removed'} {sum(~outlier_mask)} outlier"
                if feature_selection:
                    notif += f" | {'Dipilih' if lang == 'ID' else 'Selected'} {len(selected_features)} fitur terbaik"
                st.toast(notif, icon="✅")

                # --- Prediction Section ---
                st.subheader(f"🔮 {'Prediksi Penjualan' if lang == 'ID' else 'Sales Prediction'}")
                
                if f"{model_choice}_performance" in st.session_state:
                    with st.expander(f"📈 {'Performa Model' if lang == 'ID' else 'Model Performance'}: {model_choice}"):
                        perf = st.session_state[f"{model_choice}_performance"]
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("### 📊 Performa Pelatihan" if lang == "ID" else "### 📊 Training Performance")
                            st.metric("MAE", f"{perf['train_mae']:.2f}")
                            st.metric("RMSE", f"{perf['train_rmse']:.2f}")
                            st.metric("R²", f"{perf['train_r2']:.3f}")
                            st.metric("MAPE", f"{perf['train_mape']:.2f}%")

                        with col2:
                            st.markdown("### 🎯 Performa Validasi" if lang == "ID" else "### 🎯 Validation Performance")
                            st.metric("MAE", f"{perf['test_mae']:.2f}")
                            st.metric("RMSE", f"{perf['test_rmse']:.2f}")
                            st.metric("R²", f"{perf['test_r2']:.3f}")
                            st.metric("MAPE", f"{perf['test_mape']:.2f}%")                                                                                                                            
                
                        # Performance Visualization
                        fig_performance = go.Figure()
                        
                        # Add training predictions
                        fig_performance.add_trace(go.Scatter(
                            x=y_train_eval,
                            y=y_pred_train,
                            mode='markers',
                            name='Training',
                            opacity=0.6,
                            marker=dict(color='blue', size=6)
                        ))
                        
                        # Add test predictions
                        fig_performance.add_trace(go.Scatter(
                            x=y_test_eval,
                            y=y_pred_test,
                            mode='markers',
                            name='Validation',
                            opacity=0.8,
                            marker=dict(color='red', size=8)
                        ))
                        
                        # Add perfect prediction line
                        max_val = max(max(y_train_eval.max(), y_test_eval.max()), 
                                    max(y_pred_train.max(), y_pred_test.max()))
                        fig_performance.add_trace(go.Scatter(
                            x=[0, max_val],
                            y=[0, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='gray')
                        ))
                        
                        fig_performance.update_layout(
                            title='Model Performance: Actual vs Predicted',
                            xaxis_title='Actual Sales',
                            yaxis_title='Predicted Sales',
                            template=plotly_template,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_performance, use_container_width=True)
                
                # Feature Importance (if available)
                if hasattr(trained_model, 'feature_importances_'):
                    importances = trained_model.feature_importances_
                elif hasattr(trained_model, 'estimators_'):
                    # For ensemble models
                    importances = np.mean([est.feature_importances_ for est in trained_model.estimators_], axis=0)
                else:
                    importances = None
                
                if importances is not None:
                    with st.expander("🔍 Top 15 Fitur Penting" if lang == "ID" else "🔍 Top 15 Feature Importances", expanded=False):
                        feature_importance_df = pd.DataFrame({
                            'feature': selected_features,
                            'importance': importances
                        }).sort_values('importance', ascending=True).tail(15)

                        fig_importance = px.bar(
                            feature_importance_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 15 Fitur Berdasarkan Pentingnya' if lang == 'ID' else 'Top 15 Most Important Features',
                            template=plotly_template
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)

                # Model Details
                if mode == 'Advanced':
                    with st.expander("🔍 Model Details"):
                        st.write(f"**Model Type:** {model_choice}")
                        st.write(f"**Features Selected:** {len(selected_features)}")
                        st.write(f"**Training Samples:** {len(X_train_selected)}")
                        st.write(f"**Validation Samples:** {len(X_test_selected)}")
                        st.write(f"**Log Transform:** {use_log_transform}")
                        st.write(f"**Outlier Removal:** {handle_outliers}")
                        
                        if enable_hyperparameter_tuning:
                            st.write("**Best Parameters:**")
                            if hasattr(trained_model, 'get_params'):
                                st.write("**Best Parameters:**")
                                params = trained_model.get_params()
                                for param, value in list(params.items())[:10]:  # Show first 10 params
                                    st.write(f"  - {param}: {value}")
                
                logger.info(f"Model trained successfully. Test R²: {test_r2:.3f}, Test MAPE: {test_mape:.2f}%")

    # Check if model exists
    model_exists = any(os.path.exists(path) for path in MODEL_PATHS.values())
    
    if not model_exists:
        st.warning(f"⚠️ {'Model belum dilatih. Silakan latih model terlebih dahulu.' if lang == 'ID' else 'Model not trained yet. Please train the model first.'}")
    else:
        # Load the most recent model
        try:
            # Try to load feature maps first
            if os.path.exists(FEATURE_MAP_PATH):
                saved_artifacts = joblib.load(FEATURE_MAP_PATH)
                feature_maps = saved_artifacts['feature_maps']
                all_features = saved_artifacts.get('all_features', []) 
                selected_features = saved_artifacts['selected_features']
                numeric_columns = saved_artifacts['numeric_columns']
                use_log_transform = saved_artifacts['use_log_transform']
                feature_selection = saved_artifacts['feature_selection']
                selector = saved_artifacts.get('selector')
                saved_model_choice = saved_artifacts.get('model_choice', 'Random Forest')
                if not all_features:
                    st.warning("Feature list not found in artifact, using selected_features as fallback. Please retrain model.")
                    all_features = selected_features
            else:
                st.error("Feature maps not found. Please retrain the model.")
                st.stop()
            
            # Load the corresponding model
            if os.path.exists(MODEL_PATHS[saved_model_choice]):
                loaded_model = joblib.load(MODEL_PATHS[saved_model_choice])
                loaded_scaler = joblib.load(SCALER_PATH)
            else:
                st.error(f"Model file not found for {saved_model_choice}. Please retrain the model.")
                st.stop()
            
            # Prediction Interface
            col1, col2 = st.columns(2)
            
            with col1:
                nama_produk_list = sorted(df['nama_produk'].unique())
                pred_nama_produk = st.selectbox(
                    f"{'Pilih Produk' if lang == 'ID' else 'Select Product'}",
                    nama_produk_list
                )
                
                current_year = datetime.now().year
                pred_tahun = st.selectbox(
                    f"{'Pilih Tahun' if lang == 'ID' else 'Select Year'}",
                    range(current_year, current_year + 3),
                    index=0
                )
            
            with col2:
                bulan_names = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                              'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'] if lang == 'ID' else [
                              'January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                
                pred_bulan_name = st.selectbox(
                    f"{'Pilih Bulan' if lang == 'ID' else 'Select Month'}",
                    bulan_names
                )
                pred_bulan = bulan_names.index(pred_bulan_name) + 1
            
            # Bulk Prediction Option
            st.markdown("---")
            bulk_prediction = st.checkbox(f"{'Prediksi Massal (Semua atau Beberapa Produk)' if lang == 'ID' else 'Bulk Prediction (All or Some Products)'}")

            if bulk_prediction:
                prediction_months = st.slider(
                    f"{'Jumlah Bulan ke Depan' if lang == 'ID' else 'Number of Future Months'}",
                    1, 12, 6
                )

                if lang == "ID":
                    batch_mode = st.radio("Mode Prediksi Batch", ["Semua Produk", "Pilih Produk"])
                else:
                    batch_mode = st.radio("Batch Prediction Mode", ["All Products", "Select Products"])

                # Ambil daftar semua produk
                product_list = sorted(df['nama_produk'].unique())

                # Tentukan produk yang akan diprediksi
                if batch_mode == ("Pilih Produk" if lang == "ID" else "Select Products"):
                    selected_products = st.multiselect(
                        "Pilih produk untuk diprediksi" if lang == "ID" else "Select products to predict",
                        options=product_list,
                        default=product_list[:3]
                    )
                else:
                    selected_products = product_list
            else:
                selected_products = [] # No bulk prediction, so no products selected for bulk

            def safe_feature_selection(pred_input, selected_features, context_name=""):
                # Ensure pred_input has all selected_features and in the correct order
                # Reindex to selected_features and fill any new NaNs with 0 (or appropriate default)
                pred_input_aligned = pred_input.reindex(columns=selected_features, fill_value=0)

                available_features = [col for col in selected_features if col in pred_input_aligned.columns]
                missing_features = set(selected_features) - set(available_features)

                if missing_features:
                    msg = (
                        f"⚠️ {context_name}Beberapa fitur tidak tersedia: {missing_features}"
                        if lang == "ID" else
                        f"⚠️ {context_name}Some features are missing: {missing_features}"
                    )
                    st.warning(msg)

                return pred_input_aligned[available_features]

            # Make Prediction
            if st.button(f"{'🔮 Prediksi Sekarang' if lang == 'ID' else '🔮 Predict Now'}"):
                with st.spinner(f"{'Membuat prediksi...' if lang == 'ID' else 'Making prediction...'}"):
                    if bulk_prediction:
                        # Bulk prediction for all products
                        bulk_results = []
                        
                        for month_offset in range(prediction_months):
                            target_date = datetime(pred_tahun, pred_bulan, 1) + relativedelta(months=month_offset)
                            
                            for product in selected_products:
                                try:
                                    pred_input, pred_harga = create_prediction_input(
                                        df, df_agg, loaded_scaler, feature_maps, kategori_mapping,
                                        product, target_date.year, target_date.month,
                                        harga_satuan_per_produk, all_features, numeric_columns
                                    )

                                    # --- FIX: Align columns and select features correctly ---
                                    pred_input_aligned = pred_input.reindex(columns=all_features, fill_value=0)

                                    if feature_selection and selector is not None:
                                        prediction_data = selector.transform(pred_input_aligned)
                                    else:
                                        prediction_data = pred_input_aligned[selected_features]

                                    pred_result = loaded_model.predict(prediction_data)[0]

                                    if use_log_transform:
                                        pred_result = np.expm1(pred_result)
                                    
                                    pred_result = max(0, round(pred_result))

                                    bulk_results.append({
                                        'Produk': product,
                                        'Tahun': target_date.year,
                                        'Bulan': target_date.month,
                                        'Bulan_Nama': bulan_names[target_date.month - 1],
                                        'Prediksi_Jumlah': pred_result,
                                        'Harga_Satuan': pred_harga,
                                        'Prediksi_Revenue': pred_result * pred_harga
                                    })
                                except Exception as e:
                                    st.error(f"❌ Error predicting for {product}: {str(e)}")
                                    continue
                        
                        if bulk_results:
                            bulk_df = pd.DataFrame(bulk_results)
                            
                            # Display summary
                            st.success(f"✅ {'Prediksi massal berhasil!' if lang == 'ID' else 'Bulk prediction successful!'}")
                            
                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                total_predicted_sales = bulk_df['Prediksi_Jumlah'].sum()
                                st.metric(
                                    f"{'Total Prediksi Penjualan' if lang == 'ID' else 'Total Predicted Sales'}", 
                                    f"{total_predicted_sales:,.0f}"
                                )
                            
                            with col2:
                                total_predicted_revenue = bulk_df['Prediksi_Revenue'].sum()
                                st.metric(
                                    f"{'Total Prediksi Pendapatan' if lang == 'ID' else 'Total Predicted Revenue'}", 
                                    f"Rp {total_predicted_revenue:,.0f}"
                                )
                            
                            with col3:
                                avg_monthly_sales = bulk_df.groupby(['Tahun', 'Bulan'])['Prediksi_Jumlah'].sum().mean()
                                st.metric(
                                    f"{'Rata-rata Penjualan Bulanan' if lang == 'ID' else 'Average Monthly Sales'}", 
                                    f"{avg_monthly_sales:.0f}"
                                )
                            
                            # Visualization
                            monthly_summary = bulk_df.groupby(['Tahun', 'Bulan', 'Bulan_Nama']).agg({
                                'Prediksi_Jumlah': 'sum',
                                'Prediksi_Revenue': 'sum'
                            }).reset_index()
                            
                            monthly_summary['Period'] = monthly_summary['Bulan_Nama'] + ' ' + monthly_summary['Tahun'].astype(str)
                            
                            fig_bulk = go.Figure()
                            
                            fig_bulk.add_trace(go.Bar(
                                x=monthly_summary['Period'],
                                y=monthly_summary['Prediksi_Jumlah'],
                                name='Predicted Sales',
                                yaxis='y',
                                marker_color='lightblue'
                            ))
                            
                            fig_bulk.add_trace(go.Scatter(
                                x=monthly_summary['Period'],
                                y=monthly_summary['Prediksi_Revenue'],
                                mode='lines+markers',
                                name='Predicted Revenue',
                                yaxis='y2',
                                line=dict(color='red', width=3)
                            ))
                            
                            fig_bulk.update_layout(
                                title='Monthly Sales and Revenue Prediction',
                                xaxis_title='Month',
                                yaxis=dict(title='Sales Quantity', side='left'),
                                yaxis2=dict(title='Revenue (Rp)', side='right', overlaying='y'),
                                template=plotly_template,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_bulk, use_container_width=True)
                            
                            # Product-wise breakdown
                            product_summary = bulk_df.groupby('Produk').agg({
                                'Prediksi_Jumlah': 'sum',
                                'Prediksi_Revenue': 'sum'
                            }).reset_index().sort_values('Prediksi_Revenue', ascending=False)
                            
                            fig_product = px.bar(
                                product_summary.head(10),
                                x='Produk',
                                y='Prediksi_Revenue',
                                title='Top 10 Products by Predicted Revenue',
                                template=plotly_template
                            )
                            fig_product.update_layout(xaxis_tickangle=45)
                            st.plotly_chart(fig_product, use_container_width=True)
                            
                            # Detailed table
                            st.subheader(f"📊 {'Tabel Detail Prediksi' if lang == 'ID' else 'Detailed Prediction Table'}")
                            
                            # Format the display
                            display_df = bulk_df.copy()
                            display_df['Prediksi_Revenue'] = display_df['Prediksi_Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                            display_df = display_df.rename(columns={
                                'Produk': 'Product' if lang == 'EN' else 'Produk',
                                'Tahun': 'Year' if lang == 'EN' else 'Tahun',
                                'Bulan_Nama': 'Month' if lang == 'EN' else 'Bulan',
                                'Prediksi_Jumlah': 'Predicted Quantity' if lang == 'EN' else 'Prediksi Jumlah',
                                'Harga_Satuan': 'Unit Price' if lang == 'EN' else 'Harga Satuan',
                                'Prediksi_Revenue': 'Predicted Revenue' if lang == 'EN' else 'Prediksi Pendapatan'
                            })
                            
                            st.dataframe(display_df.drop(['Bulan'], axis=1), use_container_width=True)
                            
                            # Download option
                            csv = bulk_df.to_csv(index=False)
                            st.download_button(
                                label=f"{'📥 Download Hasil Prediksi' if lang == 'ID' else '📥 Download Prediction Results'}",
                                data=csv,
                                file_name=f"bulk_prediction_{pred_tahun}_{pred_bulan}.csv",
                                mime="text/csv"
                            )
                    
                    else:
                        # Single prediction
                        try:
                            # Create the feature set for the prediction
                            pred_input, pred_harga = create_prediction_input(
                                df, df_agg, loaded_scaler, feature_maps, kategori_mapping,
                                pred_nama_produk, pred_tahun, pred_bulan,
                                harga_satuan_per_produk, all_features, numeric_columns
                            )

                            # --- FIX: Align columns and select features correctly ---
                            # Ensure the input has the exact same columns as the training data
                            pred_input_aligned = pred_input.reindex(columns=all_features, fill_value=0)

                            # Apply feature selection if it was used during training
                            if feature_selection and selector is not None:
                                # The selector returns a NumPy array of the selected features
                                prediction_data = selector.transform(pred_input_aligned)
                            else:
                                # No selector, the model expects a DataFrame with all features in the correct order
                                prediction_data = pred_input_aligned[selected_features]
                            
                            # Make the prediction
                            pred_result = loaded_model.predict(prediction_data)[0]
                            
                            # Inverse transform if log was used
                            if use_log_transform:
                                pred_result = np.expm1(pred_result)
                            
                            pred_result = max(0, round(pred_result))
                            pred_revenue = pred_result * pred_harga
                            
                            # Historical context
                            historical_data = df_agg[
                                (df_agg['nama_produk'] == pred_nama_produk) & 
                                (df_agg['bulan'] == pred_bulan)
                            ]
                            
                            if not historical_data.empty:
                                historical_avg = historical_data['jumlah'].mean()
                                historical_std = historical_data['jumlah'].std()
                                
                                # Calculate confidence interval (approximate)
                                confidence_lower = max(0, pred_result - 1.96 * historical_std)
                                confidence_upper = pred_result + 1.96 * historical_std
                                
                                # Trend analysis
                                if len(historical_data) > 1:
                                    recent_trend = historical_data.tail(3)['jumlah'].mean()
                                    older_trend = historical_data.head(3)['jumlah'].mean()
                                    trend_direction = "📈 Naik" if recent_trend > older_trend else "📉 Turun" if recent_trend < older_trend else "➡️ Stabil"
                                    if lang == 'EN':
                                        trend_direction = trend_direction.replace("Naik", "Rising").replace("Turun", "Falling").replace("Stabil", "Stable")
                                else:
                                    trend_direction = "➡️ Insufficient data"
                            else:
                                historical_avg = 0
                                confidence_lower = pred_result * 0.8
                                confidence_upper = pred_result * 1.2
                                trend_direction = "❓ No historical data"
                            
                            # Display results
                            st.success(f"✅ {'Prediksi berhasil dibuat!' if lang == 'ID' else 'Prediction generated successfully!'}")
                            
                            # Main prediction results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    f"{'Prediksi Penjualan' if lang == 'ID' else 'Predicted Sales'}", 
                                    f"{pred_result:,.0f}",
                                    delta=f"{pred_result - historical_avg:.0f}" if historical_avg > 0 else None
                                )
                            
                            with col2:
                                st.metric(
                                    f"{'Prediksi Pendapatan' if lang == 'ID' else 'Predicted Revenue'}", 
                                    f"Rp {pred_revenue:,.0f}"
                                )
                            
                            with col3:
                                st.metric(
                                    f"{'Tren Historis' if lang == 'ID' else 'Historical Trend'}", 
                                    trend_direction
                                )
                            
                            # Confidence interval
                            st.markdown("---")
                            st.subheader(f"📊 {'Analisis Prediksi' if lang == 'ID' else 'Prediction Analysis'}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**{'Rentang Prediksi (95% Confidence)' if lang == 'ID' else 'Prediction Range (95% Confidence)'}:**")
                                st.write(f"{'Minimum' if lang == 'ID' else 'Minimum'}: {confidence_lower:.0f}")
                                st.write(f"{'Maksimum' if lang == 'ID' else 'Maximum'}: {confidence_upper:.0f}")
                                
                                if historical_avg > 0:
                                    deviation = abs(pred_result - historical_avg) / historical_avg * 100
                                    st.write(f"{'Deviasi dari rata-rata historis' if lang == 'ID' else 'Deviation from historical average'}: {deviation:.1f}%")
                            
                            with col2:
                                st.markdown(f"**{'Detail Prediksi' if lang == 'ID' else 'Prediction Details'}:**")
                                st.write(f"{'Produk' if lang == 'ID' else 'Product'}: {pred_nama_produk}")
                                st.write(f"{'Periode' if lang == 'ID' else 'Period'}: {bulan_names[pred_bulan-1]} {pred_tahun}")
                                st.write(f"{'Harga Satuan' if lang == 'ID' else 'Unit Price'}: Rp {pred_harga:,.0f}")
                                st.write(f"{'Model' if lang == 'ID' else 'Model'}: {saved_model_choice}")
                            
                            # Historical comparison chart
                            if not historical_data.empty:
                                fig_history = go.Figure()
                                
                                # Historical data
                                fig_history.add_trace(go.Scatter(
                                    x=historical_data['tahun'],
                                    y=historical_data['jumlah'],
                                    mode='lines+markers',
                                    name='Historical Sales',
                                    line=dict(color='blue')
                                ))
                                
                                # Prediction point
                                fig_history.add_trace(go.Scatter(
                                    x=[pred_tahun],
                                    y=[pred_result],
                                    mode='markers',
                                    name='Prediction',
                                    marker=dict(color='red', size=12, symbol='star')
                                ))
                                
                                # Confidence interval
                                fig_history.add_trace(go.Scatter(
                                    x=[pred_tahun, pred_tahun],
                                    y=[confidence_lower, confidence_upper],
                                    mode='lines',
                                    name='95% Confidence Interval',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig_history.update_layout(
                                    title=f'Historical Sales vs Prediction for {pred_nama_produk} (Month {pred_bulan})',
                                    xaxis_title='Year',
                                    yaxis_title='Sales Quantity',
                                    template=plotly_template
                                )
                                
                                st.plotly_chart(fig_history, use_container_width=True)
                            
                            # Recommendations
                            st.markdown("---")
                            st.subheader(f"💡 {'Rekomendasi Bisnis' if lang == 'ID' else 'Business Recommendations'}")
                            
                            recommendations = []
                            
                            if historical_avg > 0:
                                if pred_result > historical_avg * 1.2:
                                    recommendations.append(f"{'🚀 Prediksi menunjukkan peningkatan signifikan. Pastikan stok mencukupi!' if lang == 'ID' else '🚀 Prediction shows significant increase. Ensure adequate stock!'}")
                                elif pred_result < historical_avg * 0.8:
                                    recommendations.append(f"{'📉 Prediksi menunjukkan penurunan. Pertimbangkan strategi promosi.' if lang == 'ID' else '📉 Prediction shows decrease. Consider promotional strategies.'}")
                                else:
                                    recommendations.append(f"{'✅ Prediksi stabil sesuai tren historis.' if lang == 'ID' else '✅ Prediction is stable according to historical trend.'}")
                            
                            if pred_result > 0:
                                stock_recommendation = pred_result * 1.1  # 10% buffer
                                recommendations.append(f"{'📦 Rekomendasi stok: ' if lang == 'ID' else '📦 Stock recommendation: '}{stock_recommendation:.0f} {'unit (termasuk 10% buffer)' if lang == 'ID' else 'units (including 10% buffer)'}")
                            
                            for rec in recommendations:
                                st.write(f"• {rec}")
                            
                            logger.info(f"Prediction made for {pred_nama_produk} in {pred_bulan}/{pred_tahun}: {pred_result}")
                            
                        except Exception as e:
                            st.error(f"❌ {'Terjadi kesalahan saat membuat prediksi:' if lang == 'ID' else 'Error occurred while making prediction:'} {str(e)}")
                            logger.error(f"Prediction error: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ {'Terjadi kesalahan saat memuat model:' if lang == 'ID' else 'Error occurred while loading model:'} {str(e)}")
            logger.error(f"Model loading error: {str(e)}")
