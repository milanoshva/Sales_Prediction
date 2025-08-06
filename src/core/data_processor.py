import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='app.txt', format='%(asctime)s - %(levelname)s - %(message)s')

# Fungsi pembersihan mata uang yang telah diperbaiki
def clean_currency(value):
    """Converts a currency string (e.g., 'Rp10.000,50') to a float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # Menghapus 'Rp', spasi, dan titik sebagai pemisah ribuan, lalu mengganti koma dengan titik desimal
            return float(value.replace('Rp', '').strip().replace('.', '').replace(',', '.'))
        except (ValueError, AttributeError) as e: # <-- PERBAIKAN: "as e" ditambahkan di sini
            # Log error jika terjadi kegagalan konversi
            logging.error(f"Failed to clean currency value '{value}': {e}", exc_info=True)
            return np.nan # Mengembalikan NaN jika gagal
    return np.nan

def process_data(df):
    """
    Cleans and preprocesses the raw sales data. This is the central processing function.
    """
    # 1. Standarisasi nama kolom
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    # 2. Pastikan kolom esensial ada
    required_columns = {
        'waktu': 'object', 'nama_produk': 'object', 'jumlah': 'float64',
        'harga_satuan': 'float64', 'harga': 'float64', 'harga_setelah_pajak': 'float64',
        'total_pembayaran': 'float64', 'kategori_produk': 'object', 'id_transaksi': 'object',
        'tipe_pesanan': 'object', 'metode_pembayaran': 'object'
    }
    for col, dtype in required_columns.items():
        if col not in df.columns:
            if 'float' in dtype or 'int' in dtype:
                df[col] = 0
            else:
                df[col] = 'N/A'

    # 3. Konversi tipe data dan pembersihan utama
    df['waktu'] = pd.to_datetime(df['waktu'], errors='coerce')

    # Bersihkan semua kolom mata uang dan numerik
    currency_cols = ['harga_satuan', 'harga', 'harga_setelah_pajak', 'total_pembayaran']
    for col in currency_cols:
        if col in df.columns:
            logging.info(f"Cleaning currency column: {col}")
            df[col] = df[col].apply(clean_currency)

    df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
    
    # Isi nilai NaN yang tersisa di kolom numerik dengan 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 4. Hapus baris dengan data kritis yang hilang
    df.dropna(subset=['waktu', 'nama_produk'], inplace=True)

    # 5. Buat fitur berbasis waktu
    if pd.api.types.is_datetime64_any_dtype(df['waktu']):
        df['tahun'] = df['waktu'].dt.year
        df['bulan'] = df['waktu'].dt.month
        df['hari_dalam_minggu'] = df['waktu'].dt.dayofweek

    logging.info(f"Data processed successfully, {len(df)} rows remaining")
    return df

# Sisa fungsi di file ini tidak perlu diubah
def create_advanced_features(df, feature_maps=None, is_training=True):
    # ... (kode tidak berubah)
    df_enhanced = df.copy()
    # Basic temporal features
    df_enhanced['bulan_sin'] = np.sin(2 * np.pi * df_enhanced['bulan'] / 12)
    df_enhanced['bulan_cos'] = np.cos(2 * np.pi * df_enhanced['bulan'] / 12)
    df_enhanced['tahun_normalized'] = (df_enhanced['tahun'] - df_enhanced['tahun'].min()) / (df_enhanced['tahun'].max() - df_enhanced['tahun'].min() + 1)
    # Quarter and season features
    df_enhanced['quarter'] = ((df_enhanced['bulan'] - 1) // 3) + 1
    df_enhanced['is_q1'] = (df_enhanced['quarter'] == 1).astype(int)
    df_enhanced['is_q2'] = (df_enhanced['quarter'] == 2).astype(int)
    df_enhanced['is_q3'] = (df_enhanced['quarter'] == 3).astype(int)
    df_enhanced['is_q4'] = (df_enhanced['quarter'] == 4).astype(int)
    # Holiday and special periods (Indonesian calendar)
    df_enhanced['is_ramadan'] = df_enhanced['bulan'].isin([3, 4]).astype(int)  # Approximate Ramadan months
    df_enhanced['is_year_end'] = df_enhanced['bulan'].isin([11, 12]).astype(int)
    df_enhanced['is_mid_year'] = df_enhanced['bulan'].isin([6, 7]).astype(int)
    # Rolling statistics for each product
    df_enhanced = df_enhanced.sort_values(['nama_produk', 'tahun', 'bulan'])
    for window in [3, 6, 12]:
        df_enhanced[f'rolling_mean_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        df_enhanced[f'rolling_std_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
        df_enhanced[f'rolling_median_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(window=window, min_periods=1).median().reset_index(0, drop=True)
    df_enhanced.fillna(0, inplace=True) # Mengisi NaN dari rolling features
    # Lag features with better handling
    for lag in range(1, 7):  # Extended lag features
        df_enhanced[f'penjualan_bulan_lalu_{lag}'] = df_enhanced.groupby('nama_produk')['jumlah'].shift(lag)
    # Trend features
    df_enhanced['trend_3m'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(window=3, min_periods=2).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(0, drop=True)
    df_enhanced['trend_6m'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(window=6, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(0, drop=True)
    # Momentum features
    df_enhanced['momentum_3m'] = df_enhanced['penjualan_bulan_lalu_1'] - df_enhanced['penjualan_bulan_lalu_3']
    df_enhanced['momentum_6m'] = df_enhanced['penjualan_bulan_lalu_1'] - df_enhanced['penjualan_bulan_lalu_6']
    # Volatility features
    df_enhanced['volatility_3m'] = df_enhanced['rolling_std_3'] / (df_enhanced['rolling_mean_3'] + 1)
    df_enhanced['volatility_6m'] = df_enhanced['rolling_std_6'] / (df_enhanced['rolling_mean_6'] + 1)
    # Price-based features
    df_enhanced['price_rank'] = df_enhanced.groupby('bulan')['harga_satuan'].rank(pct=True)
    df_enhanced['price_vs_category_avg'] = df_enhanced.groupby('kategori_produk')['harga_satuan'].transform(lambda x: (x - x.mean()) / (x.std() + 1))
    # Product performance features
    if is_training:
        # Calculate during training
        feature_maps = feature_maps or {}
        # Product popularity
        product_total_sales = df_enhanced.groupby('nama_produk')['jumlah'].sum()
        feature_maps['product_popularity'] = (product_total_sales / product_total_sales.sum()).to_dict()
        # Monthly performance patterns
        monthly_patterns = df_enhanced.groupby(['nama_produk', 'bulan'])['jumlah'].mean().to_dict()
        feature_maps['monthly_patterns'] = monthly_patterns
        # Category performance
        category_performance = df_enhanced.groupby('kategori_produk')['jumlah'].mean().to_dict()
        feature_maps['category_performance'] = category_performance
        # Seasonal multipliers
        seasonal_multipliers = df_enhanced.groupby('bulan')['jumlah'].mean()
        seasonal_multipliers = (seasonal_multipliers / seasonal_multipliers.mean()).to_dict()
        feature_maps['seasonal_multipliers'] = seasonal_multipliers
    else:
        # Use pre-calculated maps during prediction
        feature_maps = feature_maps or {}
    # Apply feature maps
    df_enhanced['product_popularity'] = df_enhanced['nama_produk'].map(feature_maps.get('product_popularity', {})).fillna(0)
    df_enhanced['monthly_pattern'] = df_enhanced.set_index(['nama_produk', 'bulan']).index.map(feature_maps.get('monthly_patterns', {})).fillna(0)
    df_enhanced['category_performance'] = df_enhanced['kategori_produk'].map(feature_maps.get('category_performance', {})).fillna(0)
    df_enhanced['seasonal_multiplier'] = df_enhanced['bulan'].map(feature_maps.get('seasonal_multipliers', {})).fillna(1)
    # YoY growth with better handling
    df_enhanced['yoy_growth'] = df_enhanced.groupby(['nama_produk', 'bulan'])['jumlah'].pct_change(periods=12).replace([np.inf, -np.inf], 0).fillna(0)
    # Interaction features
    df_enhanced['price_popularity_interaction'] = df_enhanced['harga_satuan'] * df_enhanced['product_popularity']
    df_enhanced['seasonal_price_interaction'] = df_enhanced['seasonal_multiplier'] * df_enhanced['harga_satuan']
    # Fill missing values with more sophisticated methods
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'jumlah':  # Don't fill target variable
            df_enhanced[col] = df_enhanced[col].fillna(df_enhanced[col].median())
    return (df_enhanced, feature_maps) if is_training else df_enhanced

def optimize_hyperparameters(X, y, model_name, cv_folds):
    # ... (kode tidak berubah)
    param_grid = {}
    model = None

    if model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_name == 'XGBoost':
        model = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 1.0]
        }
    elif model_name == 'LightGBM':
        model = LGBMRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [20, 31, 40]
        }
    
    if model is None:
        raise ValueError(f"Unknown model name for hyperparameter optimization: {model_name}")

    # Use RandomizedSearchCV for efficiency
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings that are sampled
        cv=tscv,
        verbose=0,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    random_search.fit(X, y)
    return random_search.best_estimator_

def create_prediction_input(df_original, df_agg, scaler, feature_maps, kategori_mapping,
                            product_name, year, month, harga_satuan_per_produk,
                            selected_features, numeric_columns):
    # ... (kode tidak berubah)
    # Create a dummy DataFrame for the single prediction point
    pred_df = pd.DataFrame([{
        'nama_produk': product_name,
        'tahun': year,
        'bulan': month,
        'jumlah': 0, # Placeholder for target
    }])

    # Add 'kategori_produk' and 'harga_satuan'
    pred_df['kategori_produk'] = pred_df['nama_produk'].map(kategori_mapping)
    pred_df['harga_satuan'] = pred_df['nama_produk'].map(harga_satuan_per_produk)
    
    # Handle cases where harga_satuan might be missing for new products
    if pred_df['harga_satuan'].isnull().any():
        pred_df['harga_satuan'].fillna(df_original['harga_satuan'].mean(), inplace=True)

    # Combine with historical data for feature engineering, ensuring correct order
    # Only take relevant columns from df_agg for feature engineering context
    df_for_features = df_agg[['nama_produk', 'tahun', 'bulan', 'jumlah', 'kategori_produk', 'harga_satuan']].copy()
    combined_df = pd.concat([df_for_features, pred_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['nama_produk', 'tahun', 'bulan']).reset_index(drop=True)

    # Apply feature engineering
    combined_df_enhanced = create_advanced_features(combined_df, feature_maps=feature_maps, is_training=False)

    # Filter for the prediction row
    pred_input_row = combined_df_enhanced[
        (combined_df_enhanced['nama_produk'] == product_name) &
        (combined_df_enhanced['tahun'] == year) &
        (combined_df_enhanced['bulan'] == month)
    ].iloc[-1:] # Use iloc[-1:] to get the last matching row, ensuring it's the one we just added/engineered
    logger.info(f"pred_input_row columns before categorical encoding: {pred_input_row.columns.tolist()}")

    # Handle categorical encoding for the prediction input
    if 'all_categories' in feature_maps:
        # Ensure pred_input_row is not empty before attempting to modify 'kategori_produk'
        if not pred_input_row.empty:
            pred_input_row['kategori_produk'] = pd.Categorical(
                pred_input_row['kategori_produk'],
                categories=feature_maps['all_categories']
            )
            pred_input_row['kategori_encoded'] = pred_input_row['kategori_produk'].cat.codes
        else:
            # If pred_input_row is empty, create a dummy row with encoded category
            # This handles cases where a product might be entirely new or not in historical data
            dummy_kategori_encoded = -1 # Default for unknown category
            if product_name in kategori_mapping:
                cat = kategori_mapping[product_name]
                if cat in feature_maps['all_categories']:
                    dummy_kategori_encoded = feature_maps['all_categories'].get_loc(cat)
            
            # Create a dummy DataFrame with the required columns for feature alignment
            # This ensures 'kategori_encoded' is present even if pred_input_row was empty
            pred_input_row = pd.DataFrame([{
                'nama_produk': product_name,
                'tahun': year,
                'bulan': month,
                'jumlah': 0, # Placeholder
                'kategori_produk': kategori_mapping.get(product_name, 'N/A'),
                'harga_satuan': harga_satuan_per_produk.get(product_name, df_original['harga_satuan'].mean()),
                'kategori_encoded': dummy_kategori_encoded
            }])
    logger.info(f"pred_input_row columns after categorical encoding: {pred_input_row.columns.tolist()}")

    # Create a new DataFrame with the selected_features and populate it
    final_pred_input = pd.DataFrame(0.0, index=pred_input_row.index, columns=selected_features)

    for col in selected_features:
        if col in pred_input_row.columns:
            final_pred_input[col] = pred_input_row[col]
        # If a selected_feature is not in pred_input_row, it remains 0.0 as initialized
    logger.info(f"final_pred_input columns before scaling: {final_pred_input.columns.tolist()}")

    # Ensure numeric_columns are actually numeric in final_pred_input before scaling
    for col in numeric_columns:
        if col in final_pred_input.columns and not pd.api.types.is_numeric_dtype(final_pred_input[col]):
            final_pred_input[col] = pd.to_numeric(final_pred_input[col], errors='coerce').fillna(0) # Fallback

    # Apply scaling
    pred_input_scaled = final_pred_input.copy()
    pred_input_scaled[numeric_columns] = scaler.transform(final_pred_input[numeric_columns])
    
    # Fill any remaining NaNs after scaling (e.g., from new features that were all NaN)
    pred_input_scaled = pred_input_scaled.fillna(0) # Or a more robust imputation
    logger.info(f"pred_input_scaled columns after scaling: {pred_input_scaled.columns.tolist()}")

    # Get the unit price for revenue calculation
    unit_price = pred_input_row['harga_satuan'].iloc[0] if not pred_input_row.empty else 0

    return pred_input_scaled, unit_price