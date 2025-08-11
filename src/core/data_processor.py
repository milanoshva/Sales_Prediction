import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import logging
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='app.txt', format='%(asctime)s - %(levelname)s - %(message)s')

# Enhanced currency cleaning function
def clean_currency(value):
    """Enhanced currency converter with better handling of edge cases."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # Remove common currency symbols and clean
            s = value.replace('Rp', '').replace('$', '').replace(',', '').strip()
            
            # Handle multiple currency values in one string
            if 'Rp' in value and value.count('Rp') > 1:
                parts = [p.strip() for p in value.split('Rp') if p.strip()]
                if parts:
                    s = parts[0]  # Take the first value
            
            # Clean dots and commas properly for Indonesian format
            if '.' in s and ',' in s:
                # Indonesian format: 1.234.567,89
                s = s.replace('.', '').replace(',', '.')
            elif '.' in s and s.count('.') > 1:
                # Multiple dots (thousands separator)
                parts = s.split('.')
                if len(parts[-1]) <= 2:  # Last part is decimal
                    s = ''.join(parts[:-1]) + '.' + parts[-1]
                else:  # All dots are thousands separators
                    s = ''.join(parts)
            
            return float(s)
        except (ValueError, IndexError, AttributeError):
            logging.warning(f"Cannot convert currency string: '{value}'. Returning NaN.")
            return np.nan
    return np.nan

def process_data(df):
    """Enhanced data processing with better handling of missing values and data types."""
    # Standardize column names
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]

    # Ensure essential columns exist
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
                df[col] = 'Unknown'

    # Convert datetime with multiple format support
    df['waktu'] = pd.to_datetime(df['waktu'], errors='coerce', infer_datetime_format=True)

    # Enhanced currency cleaning
    currency_cols = ['harga_satuan', 'harga', 'harga_setelah_pajak', 'total_pembayaran']
    for col in currency_cols:
        if col in df.columns:
            logging.info(f"Cleaning currency column: {col}")
            df[col] = df[col].apply(clean_currency)

    # Convert quantity with better error handling
    df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
    
    # Better handling of missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if col != 'jumlah':  # Don't fill target variable yet
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0)
    
    # Fill categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['waktu']:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df[col] = df[col].fillna(fill_val)

    # Remove rows with critical missing data
    df.dropna(subset=['waktu', 'nama_produk'], inplace=True)
    
    # Remove rows with zero or negative quantities (likely data errors)
    df = df[df['jumlah'] > 0]

    # Create time-based features
    if pd.api.types.is_datetime64_any_dtype(df['waktu']):
        df['tahun'] = df['waktu'].dt.year
        df['bulan'] = df['waktu'].dt.month
        df['hari_dalam_minggu'] = df['waktu'].dt.dayofweek
        df['minggu_dalam_tahun'] = df['waktu'].dt.isocalendar().week

    logging.info(f"Data processed successfully, {len(df)} rows remaining")
    return df

def create_advanced_features(df, feature_maps=None, is_training=True):
    """Enhanced feature engineering with better handling of missing historical data."""
    df_enhanced = df.copy()
    
    # Ensure data is sorted properly
    df_enhanced = df_enhanced.sort_values(['nama_produk', 'tahun', 'bulan']).reset_index(drop=True)
    
    # Enhanced temporal features
    df_enhanced['bulan_sin'] = np.sin(2 * np.pi * df_enhanced['bulan'] / 12)
    df_enhanced['bulan_cos'] = np.cos(2 * np.pi * df_enhanced['bulan'] / 12)
    
    # Normalize year properly
    year_min, year_max = df_enhanced['tahun'].min(), df_enhanced['tahun'].max()
    if year_max > year_min:
        df_enhanced['tahun_normalized'] = (df_enhanced['tahun'] - year_min) / (year_max - year_min)
    else:
        df_enhanced['tahun_normalized'] = 0
    
    # Quarter and season features
    df_enhanced['quarter'] = ((df_enhanced['bulan'] - 1) // 3) + 1
    df_enhanced['is_q1'] = (df_enhanced['quarter'] == 1).astype(int)
    df_enhanced['is_q2'] = (df_enhanced['quarter'] == 2).astype(int)
    df_enhanced['is_q3'] = (df_enhanced['quarter'] == 3).astype(int)
    df_enhanced['is_q4'] = (df_enhanced['quarter'] == 4).astype(int)
    
    # Indonesian seasonal patterns
    df_enhanced['is_ramadan'] = df_enhanced['bulan'].isin([3, 4]).astype(int)
    df_enhanced['is_year_end'] = df_enhanced['bulan'].isin([11, 12]).astype(int)
    df_enhanced['is_mid_year'] = df_enhanced['bulan'].isin([6, 7]).astype(int)
    df_enhanced['is_harvest_season'] = df_enhanced['bulan'].isin([4, 5, 9, 10]).astype(int)
    
    # Enhanced rolling statistics with better fallback values
    for window in [3, 6, 12]:
        # Rolling mean with exponential decay for missing periods
        df_enhanced[f'rolling_mean_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Rolling std with minimum threshold
        rolling_std = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=1
        ).std().reset_index(0, drop=True)
        df_enhanced[f'rolling_std_{window}'] = rolling_std.fillna(df_enhanced['jumlah'].std())
        
        # Rolling median
        df_enhanced[f'rolling_median_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=1
        ).median().reset_index(0, drop=True)
        
        # Rolling max and min for range features
        df_enhanced[f'rolling_max_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=1
        ).max().reset_index(0, drop=True)
        
        df_enhanced[f'rolling_min_{window}'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=1
        ).min().reset_index(0, drop=True)
    
    # Enhanced lag features with intelligent fallbacks
    product_means = df_enhanced.groupby('nama_produk')['jumlah'].mean().to_dict()
    
    for lag in range(1, 13):  # Extended to 12 months
        lag_col = f'penjualan_bulan_lalu_{lag}'
        df_enhanced[lag_col] = df_enhanced.groupby('nama_produk')['jumlah'].shift(lag)
        
        # Fill missing lag values with product-specific patterns
        mask = df_enhanced[lag_col].isna()
        if mask.any():
            # Use product mean adjusted by seasonal pattern
            seasonal_adj = df_enhanced.loc[mask, 'bulan'].map(
                df_enhanced.groupby('bulan')['jumlah'].mean() / df_enhanced['jumlah'].mean()
            ).fillna(1)
            
            fallback_values = df_enhanced.loc[mask, 'nama_produk'].map(product_means).fillna(
                df_enhanced['jumlah'].median()
            ) * seasonal_adj
            
            df_enhanced.loc[mask, lag_col] = fallback_values
    
    # Enhanced trend features with better error handling
    for window in [3, 6, 12]:
        def safe_trend_calc(x):
            if len(x) < 2:
                return 0
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except:
                return 0
        
        df_enhanced[f'trend_{window}m'] = df_enhanced.groupby('nama_produk')['jumlah'].rolling(
            window=window, min_periods=2
        ).apply(safe_trend_calc).reset_index(0, drop=True)
    
    # Enhanced momentum features
    df_enhanced['momentum_3m'] = (df_enhanced['penjualan_bulan_lalu_1'] - 
                                 df_enhanced['penjualan_bulan_lalu_3']).fillna(0)
    df_enhanced['momentum_6m'] = (df_enhanced['penjualan_bulan_lalu_1'] - 
                                 df_enhanced['penjualan_bulan_lalu_6']).fillna(0)
    df_enhanced['momentum_12m'] = (df_enhanced['penjualan_bulan_lalu_1'] - 
                                  df_enhanced['penjualan_bulan_lalu_12']).fillna(0)
    
    # Enhanced volatility features
    for window in [3, 6, 12]:
        mean_col = f'rolling_mean_{window}'
        std_col = f'rolling_std_{window}'
        vol_col = f'volatility_{window}m'
        
        # Coefficient of variation with minimum threshold
        df_enhanced[vol_col] = df_enhanced[std_col] / (df_enhanced[mean_col] + 1)
        df_enhanced[vol_col] = df_enhanced[vol_col].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Enhanced price-based features
    df_enhanced['price_rank'] = df_enhanced.groupby(['tahun', 'bulan'])['harga_satuan'].rank(pct=True)
    
    # Price vs category average with better handling
    price_category_stats = df_enhanced.groupby('kategori_produk')['harga_satuan'].agg(['mean', 'std']).rename(columns={'mean': 'mean_cat', 'std': 'std_cat'})
    df_enhanced = df_enhanced.merge(price_category_stats, left_on='kategori_produk', right_index=True)
    df_enhanced['price_vs_category_avg'] = (df_enhanced['harga_satuan'] - df_enhanced['mean_cat']) / (df_enhanced['std_cat'] + 1)
    df_enhanced.drop(['mean_cat', 'std_cat'], axis=1, inplace=True)
    
    # Enhanced product performance features
    if is_training:
        feature_maps = feature_maps or {}
        
        # Product popularity with recency weighting
        recent_data = df_enhanced[df_enhanced['tahun'] >= df_enhanced['tahun'].max() - 1]
        product_recent_sales = recent_data.groupby('nama_produk')['jumlah'].sum()
        total_recent_sales = product_recent_sales.sum()
        
        if total_recent_sales > 0:
            feature_maps['product_popularity'] = (product_recent_sales / total_recent_sales).to_dict()
        else:
            feature_maps['product_popularity'] = {p: 1/len(product_recent_sales) 
                                               for p in product_recent_sales.index}
        
        # Enhanced monthly patterns with confidence scores
        monthly_patterns = {}
        monthly_confidence = {}
        for (product, month), group in df_enhanced.groupby(['nama_produk', 'bulan']):
            if len(group) >= 2:  # Need at least 2 data points for reliability
                monthly_patterns[(product, month)] = group['jumlah'].mean()
                monthly_confidence[(product, month)] = min(1.0, len(group) / 12)  # Confidence based on data points
            else:
                # Fallback to overall product average for that month
                product_data = df_enhanced[df_enhanced['nama_produk'] == product]
                monthly_patterns[(product, month)] = product_data['jumlah'].mean()
                monthly_confidence[(product, month)] = 0.3  # Low confidence
        
        feature_maps['monthly_patterns'] = monthly_patterns
        feature_maps['monthly_confidence'] = monthly_confidence
        
        # Enhanced category performance
        category_performance = df_enhanced.groupby('kategori_produk')['jumlah'].mean().to_dict()
        feature_maps['category_performance'] = category_performance
        
        # Enhanced seasonal multipliers with trend adjustment
        seasonal_base = df_enhanced.groupby('bulan')['jumlah'].mean()
        overall_mean = df_enhanced['jumlah'].mean()
        seasonal_multipliers = (seasonal_base / overall_mean).to_dict()
        
        # Add year-over-year trend to seasonal multipliers
        yearly_trend = df_enhanced.groupby('tahun')['jumlah'].mean().pct_change().mean()
        if not pd.isna(yearly_trend) and yearly_trend != 0:
            current_year = df_enhanced['tahun'].max()
            years_from_base = max(0, current_year - df_enhanced['tahun'].min())
            trend_adjustment = 1 + (yearly_trend * years_from_base)
            seasonal_multipliers = {k: v * trend_adjustment for k, v in seasonal_multipliers.items()}
        
        feature_maps['seasonal_multipliers'] = seasonal_multipliers
        
        # Store global statistics for fallback
        feature_maps['global_stats'] = {
            'overall_mean': overall_mean,
            'overall_std': df_enhanced['jumlah'].std(),
            'overall_median': df_enhanced['jumlah'].median()
        }
        
    else:
        # Use pre-calculated maps during prediction
        feature_maps = feature_maps or {}
    
    # Apply feature maps with enhanced fallbacks
    df_enhanced['product_popularity'] = df_enhanced['nama_produk'].map(
        feature_maps.get('product_popularity', {})
    ).fillna(feature_maps.get('global_stats', {}).get('overall_mean', 1) / 
             len(feature_maps.get('product_popularity', {1: 1})))
    
    # Enhanced monthly pattern with confidence weighting
    monthly_patterns = feature_maps.get('monthly_patterns', {})
    monthly_confidence = feature_maps.get('monthly_confidence', {})
    
    def get_monthly_pattern_value(row):
        key = (row['nama_produk'], row['bulan'])
        if key in monthly_patterns:
            pattern_value = monthly_patterns[key]
            confidence = monthly_confidence.get(key, 0.5)
            
            # Blend with global average based on confidence
            global_avg = feature_maps.get('global_stats', {}).get('overall_mean', pattern_value)
            return pattern_value * confidence + global_avg * (1 - confidence)
        else:
            # Fallback to seasonal pattern
            seasonal_mult = feature_maps.get('seasonal_multipliers', {}).get(row['bulan'], 1)
            product_pop = feature_maps.get('product_popularity', {}).get(row['nama_produk'], 1)
            global_avg = feature_maps.get('global_stats', {}).get('overall_mean', 10)
            return global_avg * seasonal_mult * product_pop
    
    df_enhanced['monthly_pattern'] = df_enhanced.apply(get_monthly_pattern_value, axis=1)
    
    # Apply other feature maps
    df_enhanced['category_performance'] = df_enhanced['kategori_produk'].map(
        feature_maps.get('category_performance', {})
    ).fillna(feature_maps.get('global_stats', {}).get('overall_mean', 1))
    
    df_enhanced['seasonal_multiplier'] = df_enhanced['bulan'].map(
        feature_maps.get('seasonal_multipliers', {})
    ).fillna(1)
    
    # Enhanced YoY growth with better handling
    df_enhanced['yoy_growth'] = df_enhanced.groupby(['nama_produk', 'bulan'])['jumlah'].pct_change(periods=12)
    df_enhanced['yoy_growth'] = df_enhanced['yoy_growth'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Clip extreme values
    yoy_q99 = df_enhanced['yoy_growth'].quantile(0.99)
    yoy_q01 = df_enhanced['yoy_growth'].quantile(0.01)
    df_enhanced['yoy_growth'] = df_enhanced['yoy_growth'].clip(yoy_q01, yoy_q99)
    
    # Enhanced interaction features
    df_enhanced['price_popularity_interaction'] = (df_enhanced['harga_satuan'] * 
                                                  df_enhanced['product_popularity'])
    df_enhanced['seasonal_price_interaction'] = (df_enhanced['seasonal_multiplier'] * 
                                               df_enhanced['harga_satuan'])
    df_enhanced['trend_momentum_interaction'] = (df_enhanced['trend_3m'] * 
                                               df_enhanced['momentum_3m'])
    
    # Growth rate features
    df_enhanced['growth_rate_3m'] = ((df_enhanced['rolling_mean_3'] - 
                                     df_enhanced['penjualan_bulan_lalu_3']) / 
                                    (df_enhanced['penjualan_bulan_lalu_3'] + 1)).fillna(0)
    
    df_enhanced['growth_rate_6m'] = ((df_enhanced['rolling_mean_6'] - 
                                     df_enhanced['penjualan_bulan_lalu_6']) / 
                                    (df_enhanced['penjualan_bulan_lalu_6'] + 1)).fillna(0)
    
    # Market share features (relative to category)
    category_totals = df_enhanced.groupby(['kategori_produk', 'tahun', 'bulan'])['jumlah'].transform('sum')
    df_enhanced['market_share_in_category'] = df_enhanced['jumlah'] / (category_totals + 1)
    
    # Final cleanup of all numeric features
    numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'jumlah':  # Don't modify target variable
            # Replace inf/-inf with reasonable values
            col_median = df_enhanced[col].median()
            df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf], col_median)
            
            # Fill remaining NaN values
            df_enhanced[col] = df_enhanced[col].fillna(col_median)
            
            # Clip extreme outliers (beyond 99.9th percentile)
            if df_enhanced[col].std() > 0:  # Only if there's variation
                q999 = df_enhanced[col].quantile(0.999)
                q001 = df_enhanced[col].quantile(0.001)
                df_enhanced[col] = df_enhanced[col].clip(q001, q999)
    
    return df_enhanced, feature_maps

# Rest of the functions remain the same but with enhanced parameter grids
def optimize_hyperparameters(X, y, model_name, cv_folds):
    """Enhanced hyperparameter optimization with better parameter ranges."""
    param_grid = {}
    model = None

    if model_name == 'Random Forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_name == 'XGBoost':
        model = XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse')
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
    elif model_name == 'LightGBM':
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [10, 15, 20, -1],
            'learning_rate': [0.05, 0.1, 0.15, 0.2],
            'num_leaves': [31, 50, 70, 100],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0],
            'bagging_freq': [0, 5, 10],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
    
    if model is None:
        raise ValueError(f"Unknown model name for hyperparameter optimization: {model_name}")

    # Use RandomizedSearchCV with more iterations for better results
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Increased from 10
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
                            all_features, numeric_columns):
    """Enhanced prediction input creation with better feature alignment."""
    # Create prediction DataFrame
    pred_df = pd.DataFrame([{
        'nama_produk': product_name,
        'tahun': year,
        'bulan': month,
        'jumlah': 0,  # Placeholder
    }])

    # Add required fields
    pred_df['kategori_produk'] = pred_df['nama_produk'].map(kategori_mapping)
    pred_df['harga_satuan'] = pred_df['nama_produk'].map(harga_satuan_per_produk)
    
    # Handle missing values with global averages
    if pred_df['harga_satuan'].isnull().any():
        pred_df['harga_satuan'].fillna(df_original['harga_satuan'].mean(), inplace=True)
    if pred_df['kategori_produk'].isnull().any():
        pred_df['kategori_produk'].fillna('Unknown', inplace=True)

    # Combine with historical data for proper feature engineering
    historical_columns = ['nama_produk', 'tahun', 'bulan', 'jumlah', 'kategori_produk', 'harga_satuan']
    df_for_features = df_agg[historical_columns].copy()
    
    # Ensure prediction row has all required columns
    for col in historical_columns:
        if col not in pred_df.columns:
            pred_df[col] = 0 if df_for_features[col].dtype in ['int64', 'float64'] else 'Unknown'
    
    combined_df = pd.concat([df_for_features, pred_df], ignore_index=True)
    combined_df = combined_df.sort_values(by=['nama_produk', 'tahun', 'bulan']).reset_index(drop=True)

    # Apply enhanced feature engineering
    combined_df_enhanced, _ = create_advanced_features(combined_df, feature_maps=feature_maps, is_training=False)

    # Extract the prediction row
    pred_input_row = combined_df_enhanced[
        (combined_df_enhanced['nama_produk'] == product_name) &
        (combined_df_enhanced['tahun'] == year) &
        (combined_df_enhanced['bulan'] == month)
    ].iloc[-1:].copy()

    if pred_input_row.empty:
        raise ValueError(f"Could not create prediction input for {product_name} {year}-{month}")

    # Create final prediction input with proper feature alignment
    final_pred_input = pd.DataFrame(columns=all_features, index=pred_input_row.index)

    # Populate features that exist
    for col in all_features:
        if col in pred_input_row.columns:
            final_pred_input[col] = pred_input_row[col]
        else:
            # Use intelligent defaults for missing features
            if 'rolling_mean' in col or 'monthly_pattern' in col:
                # Use product popularity as proxy
                product_pop = feature_maps.get('product_popularity', {}).get(product_name, 1)
                global_mean = feature_maps.get('global_stats', {}).get('overall_mean', 10)
                final_pred_input[col] = global_mean * product_pop
            elif 'seasonal_multiplier' in col:
                final_pred_input[col] = feature_maps.get('seasonal_multipliers', {}).get(month, 1)
            else:
                final_pred_input[col] = 0

    # Fill any remaining NaN values
    final_pred_input = final_pred_input.fillna(0)

    # Apply scaling to numeric columns
    numeric_cols_to_scale = [col for col in numeric_columns if col in final_pred_input.columns]
    if numeric_cols_to_scale and scaler is not None:
        try:
            final_pred_input[numeric_cols_to_scale] = scaler.transform(final_pred_input[numeric_cols_to_scale])
        except Exception as e:
            logging.warning(f"Scaling failed: {e}. Using unscaled features.")

    # Get unit price for revenue calculation
    unit_price = pred_input_row['harga_satuan'].iloc[0] if not pred_input_row.empty else 0

    return final_pred_input, unit_price