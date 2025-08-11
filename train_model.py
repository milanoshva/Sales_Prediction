import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Import your custom modules (adjust paths as needed)
try:
    from src.core.data_processor import create_advanced_features, optimize_hyperparameters, process_data
except ImportError:
    print("Warning: Could not import custom modules. Make sure the paths are correct.")
    # You might need to add fallback implementations or adjust imports

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Konstanta & Konfigurasi Pelatihan ---
MODEL_DIR = "models"
DATA_DIR = "data"
TRAINING_DATA_FILE = "transaksi_haluna_2023-2024.csv"  # Ganti dengan nama file Anda

# Konfigurasi yang bisa diubah
CONFIG = {
    'cv_folds': 3,
    'outlier_threshold': 3.0,
    'k_best_features': 20,
    'use_log_transform': True,
    'use_feature_selection': False,
    'enable_hyperparameter_tuning': True,
    'handle_outliers': True,
    'test_size': 0.2  # Proporsi data untuk testing
}

MODEL_PATHS = {
    'Random Forest': os.path.join(MODEL_DIR, 'model_rf_jumlah.pkl'),
    'XGBoost': os.path.join(MODEL_DIR, 'model_xgb_jumlah.pkl'),
    'LightGBM': os.path.join(MODEL_DIR, 'model_lgb_jumlah.pkl'),
    'Gabungan': os.path.join(MODEL_DIR, 'model_ensemble_jumlah.pkl')
}
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
ARTIFACTS_PATH = os.path.join(MODEL_DIR, 'training_artifacts.pkl') # Mengganti FEATURE_MAP_PATH

def load_and_preprocess_data(file_path):
    """
    Memuat dan memproses data mentah menjadi format yang siap untuk pelatihan.
    Termasuk deteksi delimiter otomatis untuk file CSV.
    """
    logger.info(f"Memuat data dari {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"File tidak ditemukan: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error membaca file: {str(e)}")
        raise

    df = None
    for sep in [',', ';', '\t']:
        try:
            temp_df = pd.read_csv(StringIO(content), sep=sep)
            if temp_df.shape[1] > 5: # Heuristik: harus ada lebih dari 5 kolom
                df = temp_df
                logger.info(f"Berhasil membaca CSV dengan pemisah: '{repr(sep)}'")
                break
        except Exception:
            continue
    
    if df is None:
        raise ValueError("Gagal membaca file CSV dengan pemisah yang umum (, ; \t).")

    logger.info(f"Data mentah berhasil dimuat: {len(df)} baris")
    
    # Gunakan fungsi process_data dari core untuk pembersihan awal
    df_processed = process_data(df.copy())
    logger.info(f"Data setelah pembersihan awal: {len(df_processed)} baris")

    # Agregasi data ke level bulanan
    logger.info("Mengagregasi data ke level bulanan...")
    df_agg = df_processed.groupby(['tahun', 'bulan', 'nama_produk']).agg(
        jumlah=('jumlah', 'sum'),
        harga=('harga', 'sum'),
        kategori_produk=('kategori_produk', 'first'),
        harga_satuan=('harga_satuan', 'mean')
    ).reset_index()
    
    df_agg['harga_satuan'].fillna(df_agg['harga_satuan'].mean(), inplace=True)
    
    logger.info(f"Data teragregasi: {len(df_agg)} baris")
    return df_agg

def prepare_features_and_target(df_agg, config):
    """
    Modified version that stores product-specific feature defaults
    """
    logger.info("Membuat fitur-fitur canggih untuk pelatihan...")
    
    df_train_ready = df_agg.sort_values(['nama_produk', 'tahun', 'bulan']).copy()
    
    df_enhanced, feature_maps = create_advanced_features(df_train_ready, is_training=True)
    
    # One-hot encode 'nama_produk'
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    product_encoded = ohe.fit_transform(df_enhanced[['nama_produk']])
    product_df = pd.DataFrame(product_encoded, columns=ohe.get_feature_names_out(['nama_produk']), index=df_enhanced.index)
    df_enhanced = pd.concat([df_enhanced, product_df], axis=1)

    df_enhanced['kategori_produk'] = df_enhanced['kategori_produk'].astype('category')
    feature_maps['all_categories'] = df_enhanced['kategori_produk'].cat.categories
    df_enhanced['kategori_encoded'] = df_enhanced['kategori_produk'].cat.codes
    
    # NEW: Store product-specific defaults for rolling/lag features
    logger.info("Calculating product-specific feature defaults...")
    
    rolling_lag_features = [col for col in df_enhanced.columns 
                           if any(x in col for x in ['rolling_', 'penjualan_bulan_lalu_', 'trend_', 'momentum_', 'volatility_'])]
    
    for product in df_enhanced['nama_produk'].unique():
        product_data = df_enhanced[df_enhanced['nama_produk'] == product]
        product_defaults = {}
        
        for feature in rolling_lag_features:
            if feature in product_data.columns:
                # Use the median of the last 6 months or all available data
                recent_data = product_data[feature].dropna()
                if len(recent_data) > 0:
                    product_defaults[feature] = recent_data.median()
                else:
                    product_defaults[feature] = 0
        
        feature_maps[f"{product}_defaults"] = product_defaults
    
    # Update feature_columns to include one-hot encoded product features
    feature_columns = [col for col in df_enhanced.columns 
                      if col not in ['jumlah', 'nama_produk', 'kategori_produk', 'waktu']]
    
    logger.info(f"Jumlah fitur yang dibuat: {len(feature_columns)}")
    logger.info(f"Product defaults stored for {len(df_enhanced['nama_produk'].unique())} products")
    
    return df_enhanced, feature_maps, feature_columns, ohe


def handle_outliers_and_transform(df_enhanced, feature_columns, config):
    """
    Menangani outlier dan transformasi data.
    """
    logger.info("Menangani outliers dan transformasi data...")
    
    if config['handle_outliers']:
        z_scores = np.abs(stats.zscore(df_enhanced['jumlah']))
        outlier_mask = z_scores < config['outlier_threshold']
        records_before = len(df_enhanced)
        df_enhanced = df_enhanced[outlier_mask]
        removed_outliers = records_before - len(df_enhanced)
        logger.info(f"Menghapus {removed_outliers} outliers (Z-score > {config['outlier_threshold']})")
    
    X = df_enhanced[feature_columns].copy()
    y = df_enhanced['jumlah'].copy()
    
    if config['use_log_transform']:
        y_transformed = np.log1p(y)
        logger.info("Menerapkan transformasi log pada target")
    else:
        y_transformed = y
    
    return X, y_transformed, y

def scale_and_select_features(X, y, config):
    """
    Scaling fitur dan seleksi fitur.
    """
    logger.info("Melakukan scaling pada fitur numerik...")
    
    scaler = RobustScaler()
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    X_scaled = X.copy()
    X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    X_scaled = X_scaled.fillna(X_scaled.median())
    
    if config['use_feature_selection']:
        logger.info(f"Memilih {config['k_best_features']} fitur terbaik...")
        try:
            k_best = min(config['k_best_features'], len(X_scaled.columns))
            selector = SelectKBest(score_func=f_regression, k=k_best)
            
            if X_scaled.isnull().values.any():
                raise ValueError("X_scaled mengandung NaN setelah fillna")
            
            X_selected = selector.fit_transform(X_scaled, y)
            selected_features = [X_scaled.columns[i] for i in selector.get_support(indices=True)]
            logger.info(f"Fitur terpilih: {len(selected_features)} dari {len(X_scaled.columns)}")
            
        except Exception as e:
            logger.warning(f"Feature selection gagal: {e}. Menggunakan semua fitur.")
            X_selected = X_scaled.values
            selected_features = X_scaled.columns.tolist()
            selector = None
    else:
        X_selected = X_scaled.values
        selected_features = X_scaled.columns.tolist()
        selector = None
    
    return X_selected, selected_features, scaler, selector, numeric_columns, X_scaled.columns.tolist()

def train_models(X_train, y_train, config):
    """
    Melatih semua model (Random Forest, XGBoost, LightGBM, dan Ensemble).
    """
    logger.info("Memulai pelatihan model...")
    
    models = {}
    
    if config['enable_hyperparameter_tuning']:
        logger.info("Melakukan optimasi hyperparameter...")
        models['Random Forest'] = optimize_hyperparameters(X_train, y_train, 'Random Forest', config['cv_folds'])
        models['XGBoost'] = optimize_hyperparameters(X_train, y_train, 'XGBoost', config['cv_folds'])
        models['LightGBM'] = optimize_hyperparameters(X_train, y_train, 'LightGBM', config['cv_folds'])
    else:
        logger.info("Menggunakan parameter default...")
        models['Random Forest'] = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        models['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        models['LightGBM'] = LGBMRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    
    for name, model in models.items():
        logger.info(f"Melatih model {name}...")
        model.fit(X_train, y_train)
    
    logger.info("Melatih model ensemble...")
    ensemble_model = VotingRegressor([
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM'])
    ], weights=[0.4, 0.3, 0.3])
    
    ensemble_model.fit(X_train, y_train)
    models['Gabungan'] = ensemble_model
    
    logger.info("Pelatihan semua model selesai!")
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, config, feature_names):
    """
    Evaluasi performa semua model.
    """
    logger.info("Mengevaluasi performa model...")
    
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    results = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    for name, model in models.items():
        logger.info(f"Evaluasi model {name}...")
        
        y_pred_train = model.predict(X_train_df)
        y_pred_test = model.predict(X_test_df)
        
        if config['use_log_transform']:
            y_pred_train_orig = np.expm1(y_pred_train)
            y_pred_test_orig = np.expm1(y_pred_test)
        else:
            y_pred_train_orig = y_pred_train
            y_pred_test_orig = y_pred_test
        
        results[name] = {
            'train_mae': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'train_rmse': np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig)),
            'train_r2': r2_score(y_train_orig, y_pred_train_orig),
            'test_r2': r2_score(y_test_orig, y_pred_test_orig),
            'train_mape': mean_absolute_percentage_error(y_train_orig, y_pred_train_orig),
            'test_mape': mean_absolute_percentage_error(y_test_orig, y_pred_test_orig)
        }
        
        logger.info(f"{name} - Test R²: {results[name]['test_r2']:.3f}, Test MAPE: {results[name]['test_mape']:.2f}%")
    
    return results

def save_models_and_artifacts(models, scaler, feature_maps, all_features, selected_features, 
                             numeric_columns, selector, config, results, product_encoder):
    """
    Menyimpan semua model dan artefak yang diperlukan.
    """
    logger.info("Menyimpan model dan artefak...")
    
    for name, model in models.items():
        joblib.dump(model, MODEL_PATHS[name])
        logger.info(f"Model {name} disimpan ke {MODEL_PATHS[name]}")
    
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler disimpan ke {SCALER_PATH}")
    
    artifacts = {
        'feature_maps': feature_maps,
        'all_features_before_selection': all_features,
        'selected_features_after_selection': selected_features,
        'numeric_columns_to_scale': numeric_columns,
        'kbest_selector': selector,
        'config': config,
        'training_results': results,
        'training_timestamp': datetime.now().isoformat(),
        'product_encoder': product_encoder # Save the encoder
    }
    
    joblib.dump(artifacts, ARTIFACTS_PATH)
    logger.info(f"Artefak disimpan ke {ARTIFACTS_PATH}")

def main(custom_config=None):
    """
    Fungsi utama untuk menjalankan alur kerja pelatihan model.
    """
    config = CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    data_file_path = os.path.join(DATA_DIR, TRAINING_DATA_FILE)
    
    try:
        df_agg = load_and_preprocess_data(data_file_path)
        
        if len(df_agg) < 30:
            raise ValueError(f"Data tidak cukup untuk pelatihan. Minimal 30 records, ditemukan {len(df_agg)}")
        
        df_enhanced, feature_maps, feature_columns, ohe = prepare_features_and_target(df_agg, config)
        
        X, y_transformed, y_original = handle_outliers_and_transform(df_enhanced, feature_columns, config)
        
        logger.info("Melakukan train-test split dengan TimeSeriesSplit...")
        tscv = TimeSeriesSplit(n_splits=config['cv_folds'])
        train_idx, test_idx = list(tscv.split(X))[-1]
        
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_transformed.iloc[train_idx], y_transformed.iloc[test_idx]
        y_train_orig, y_test_orig = y_original.iloc[train_idx], y_original.iloc[test_idx]
        
        X_train, selected_features, scaler, selector, numeric_columns, all_feature_names = scale_and_select_features(
            X_train_raw.copy(), y_train.copy(), config
        )
        
        X_test_scaled = X_test_raw.copy()
        X_test_scaled[numeric_columns] = scaler.transform(X_test_raw[numeric_columns])
        X_test_scaled = X_test_scaled.fillna(X_train_raw.median())
        
        if config['use_feature_selection'] and selector is not None:
            X_test = selector.transform(X_test_scaled[all_feature_names])
        else:
            X_test = X_test_scaled[selected_features].values

        models = train_models(X_train, y_train, config)
        
        results = evaluate_models(models, X_train, X_test, y_train, y_test, 
                                y_train_orig, y_test_orig, config, selected_features)
        
        save_models_and_artifacts(models, scaler, feature_maps, all_feature_names, 
                                selected_features, numeric_columns, selector, config, results, ohe)
        
        logger.info("=== PELATIHAN SELESAI ===")
        
    except Exception as e:
        logger.error(f"Pelatihan gagal: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        main()
        print("\nPelatihan berhasil!")
    except Exception as e:
        print(f"\nPelatihan gagal. Lihat 'training.log' untuk detail. Error: {e}")
