import pandas as pd
import numpy as np
import joblib
import logging
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

# Import enhanced custom modules
try:
    from src.core.data_processor import create_advanced_features, optimize_hyperparameters, process_data
except ImportError:
    print("Warning: Could not import custom modules. Using local implementations.")

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants and paths
MODEL_DIR = "models"
DATA_DIR = "data"
TRAINING_DATA_FILE = "transaksi_haluna_2023-2024.csv"

# Enhanced configuration with better default parameters
CONFIG = {
    'cv_folds': 15,
    'outlier_threshold': 3.0,
    'k_best_features': 30,
    'use_log_transform': False,
    'use_feature_selection': True,
    'enable_hyperparameter_tuning': True,
    'handle_outliers': True,
    'test_size': 0.2,
    'min_samples_per_product': 10,
    'feature_importance_threshold': 0.001,
    'ensemble_weights': [0.25, 0.35, 0.3, 0.1],  # RF, XGB, LGB, GB weights
    'use_advanced_scaling': True,  # Use both robust and standard scaling
    'prediction_boost_factor': 1.5
}

MODEL_PATHS = {
    'Random Forest': os.path.join(MODEL_DIR, 'model_rf_jumlah.pkl'),
    'XGBoost': os.path.join(MODEL_DIR, 'model_xgb_jumlah.pkl'),
    'LightGBM': os.path.join(MODEL_DIR, 'model_lgb_jumlah.pkl'),
    'Gradient Boosting': os.path.join(MODEL_DIR, 'model_gb_jumlah.pkl'),
    'Gabungan': os.path.join(MODEL_DIR, 'model_ensemble_jumlah.pkl')
}
SCALER_PATH = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
ARTIFACTS_PATH = os.path.join(MODEL_DIR, 'training_artifacts.pkl')

def load_and_preprocess_data(file_path):
    """Enhanced data loading with better validation and preprocessing."""
    logger.info(f"Loading data from {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

    # Try different separators
    df = None
    for sep in [',', ';', '\t']:
        try:
            temp_df = pd.read_csv(StringIO(content), sep=sep)
            if temp_df.shape[1] > 5:
                df = temp_df
                logger.info(f"Successfully read CSV with separator: '{repr(sep)}'")
                break
        except Exception:
            continue
    
    if df is None:
        raise ValueError("Failed to read CSV file with common separators (, ; \\t).")

    logger.info(f"Raw data loaded: {len(df)} rows, {df.shape[1]} columns")
    
    # Enhanced data processing
    df_processed = process_data(df.copy())
    logger.info(f"Data after initial cleaning: {len(df_processed)} rows")

    # Enhanced aggregation with better handling of edge cases
    logger.info("Aggregating data to monthly level...")
    
    # Create datetime index for proper time-based aggregation
    df_processed['year_month'] = df_processed['waktu'].dt.to_period('M')
    
    df_agg = df_processed.groupby(['nama_produk', 'year_month']).agg(
        jumlah=('jumlah', 'sum'),
        harga=('harga', 'sum'),
        kategori_produk=('kategori_produk', 'first'),
        harga_satuan=('harga_satuan', 'mean'),
        tahun=('tahun', 'first'),
        bulan=('bulan', 'first'),
        total_transactions=('jumlah', 'count')  # Track number of transactions
    ).reset_index()
    
    # Convert period back to datetime for consistency
    df_agg['waktu'] = df_agg['year_month'].dt.to_timestamp()
    df_agg.drop('year_month', axis=1, inplace=True)
    
    # Fill missing price information with intelligent defaults
    for product in df_agg['nama_produk'].unique():
        product_mask = df_agg['nama_produk'] == product
        product_data = df_agg[product_mask]
        
        # Fill missing unit prices with product's historical average
        if product_data['harga_satuan'].isna().any():
            product_mean_price = product_data['harga_satuan'].dropna().mean()
            if pd.isna(product_mean_price):
                # Fallback to category average
                category = product_data['kategori_produk'].iloc[0]
                category_mean = df_agg[df_agg['kategori_produk'] == category]['harga_satuan'].mean()
                product_mean_price = category_mean if not pd.isna(category_mean) else df_agg['harga_satuan'].mean()
            
            df_agg.loc[product_mask, 'harga_satuan'] = df_agg.loc[product_mask, 'harga_satuan'].fillna(product_mean_price)
    
    # Remove products with insufficient data
    product_counts = df_agg['nama_produk'].value_counts()
    valid_products = product_counts[product_counts >= CONFIG['min_samples_per_product']].index
    df_agg = df_agg[df_agg['nama_produk'].isin(valid_products)]
    
    logger.info(f"Data after aggregation and filtering: {len(df_agg)} rows, {len(valid_products)} products")
    logger.info(f"Date range: {df_agg['waktu'].min()} to {df_agg['waktu'].max()}")
    
    return df_agg

def prepare_features_and_target(df_agg, config):
    """Enhanced feature preparation with better encoding and validation."""
    logger.info("Creating advanced features for training...")
    
    # Ensure proper sorting for time series features
    df_train_ready = df_agg.sort_values(['nama_produk', 'tahun', 'bulan']).copy()
    
    # Create enhanced features
    df_enhanced, feature_maps = create_advanced_features(df_train_ready, is_training=True)
    
    # Enhanced product encoding with frequency-based handling
    logger.info("Encoding categorical variables...")
    
    # Create product frequency features
    product_counts = df_enhanced['nama_produk'].value_counts()
    df_enhanced['product_frequency'] = df_enhanced['nama_produk'].map(product_counts)
    df_enhanced['product_frequency_rank'] = df_enhanced['product_frequency'].rank(pct=True)
    
    # One-hot encode products (with top-K approach for high cardinality)
    product_counts_sorted = product_counts.sort_values(ascending=False)
    top_products = product_counts_sorted.head(50).index  # Limit to top 50 products
    
    # Create binary features for top products
    for product in top_products:
        df_enhanced[f'is_product_{product}'] = (df_enhanced['nama_produk'] == product).astype(int)
    
    # Traditional one-hot encoding as well
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=30)
    product_encoded = ohe.fit_transform(df_enhanced[['nama_produk']])
    product_df = pd.DataFrame(
        product_encoded, 
        columns=ohe.get_feature_names_out(['nama_produk']), 
        index=df_enhanced.index
    )
    df_enhanced = pd.concat([df_enhanced, product_df], axis=1)

    # Enhanced category encoding
    df_enhanced['kategori_produk'] = df_enhanced['kategori_produk'].astype('category')
    feature_maps['all_categories'] = df_enhanced['kategori_produk'].cat.categories.tolist()
    df_enhanced['kategori_encoded'] = df_enhanced['kategori_produk'].cat.codes
    
    # Create category-based features
    category_stats = df_enhanced.groupby('kategori_produk')['jumlah'].agg(['mean', 'std', 'count'])
    df_enhanced = df_enhanced.merge(
        category_stats, 
        left_on='kategori_produk', 
        right_index=True, 
        suffixes=('', '_category_stat')
    )
    
    # Rename columns to avoid conflicts
    df_enhanced.rename(columns={
        'mean': 'category_mean_sales',
        'std': 'category_std_sales', 
        'count': 'category_product_count'
    }, inplace=True)
    
    # Create relative performance vs category
    df_enhanced['performance_vs_category'] = (
        df_enhanced['jumlah'] / (df_enhanced['category_mean_sales'] + 1)
    )
    
    # Store product-specific feature defaults (enhanced)
    logger.info("Calculating enhanced product-specific feature defaults...")
    
    rolling_lag_features = [col for col in df_enhanced.columns 
                           if any(x in col for x in ['rolling_', 'penjualan_bulan_lalu_', 
                                                    'trend_', 'momentum_', 'volatility_'])]
    
    # Calculate more sophisticated defaults
    for product in df_enhanced['nama_produk'].unique():
        product_data = df_enhanced[df_enhanced['nama_produk'] == product].copy()
        product_defaults = {}
        
        # Calculate seasonal patterns for the product
        monthly_patterns = product_data.groupby('bulan')['jumlah'].mean().to_dict()
        
        for feature in rolling_lag_features:
            if feature in product_data.columns:
                feature_values = product_data[feature].dropna()
                
                if len(feature_values) > 0:
                    # Use multiple statistics for more robust defaults
                    product_defaults[feature] = {
                        'median': feature_values.median(),
                        'mean': feature_values.mean(),
                        'q75': feature_values.quantile(0.75),
                        'recent': feature_values.tail(3).mean() if len(feature_values) >= 3 else feature_values.mean()
                    }
                else:
                    # Fallback based on global patterns and product characteristics
                    product_pop = feature_maps.get('product_popularity', {}).get(product, 1)
                    global_mean = feature_maps.get('global_stats', {}).get('overall_mean', 10)
                    
                    fallback_value = global_mean * product_pop
                    product_defaults[feature] = {
                        'median': fallback_value,
                        'mean': fallback_value,
                        'q75': fallback_value * 1.2,
                        'recent': fallback_value
                    }
        
        # Store monthly patterns for this product
        product_defaults['monthly_patterns'] = monthly_patterns
        feature_maps[f"{product}_defaults"] = product_defaults
    
    # Identify feature columns (exclude non-feature columns)
    exclude_cols = ['jumlah', 'nama_produk', 'kategori_produk', 'waktu', 'year_month']
    feature_columns = [col for col in df_enhanced.columns if col not in exclude_cols]
    
    logger.info(f"Total features created: {len(feature_columns)}")
    logger.info(f"Enhanced product defaults stored for {len(df_enhanced['nama_produk'].unique())} products")
    
    return df_enhanced, feature_maps, feature_columns, ohe

def handle_outliers_and_transform(df_enhanced, feature_columns, config):
    """Enhanced outlier handling with better preservation of valid high values."""
    logger.info("Handling outliers and data transformation...")
    
    if config['handle_outliers']:
        # Use IQR method instead of Z-score for better handling of skewed data
        Q1 = df_enhanced['jumlah'].quantile(0.25)
        Q3 = df_enhanced['jumlah'].quantile(0.75)
        IQR = Q3 - Q1
        
        # More conservative outlier detection
        lower_bound = Q1 - 2.0 * IQR  # Less aggressive than 1.5 * IQR
        upper_bound = Q3 + 2.0 * IQR
        
        outlier_mask = (df_enhanced['jumlah'] >= lower_bound) & (df_enhanced['jumlah'] <= upper_bound)
        records_before = len(df_enhanced)
        df_enhanced = df_enhanced[outlier_mask]
        removed_outliers = records_before - len(df_enhanced)
        logger.info(f"Removed {removed_outliers} outliers using IQR method (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
    
    X = df_enhanced[feature_columns].copy()
    y = df_enhanced['jumlah'].copy()
    
    # Avoid log transform as it can suppress predictions
    if config.get('use_log_transform', False):
        y_transformed = np.log1p(y)
        logger.info("Applied log transformation to target")
    else:
        y_transformed = y.copy()
        logger.info("Using original scale for target variable")
    
    return X, y_transformed, y

def scale_and_select_features(X, y, config):
    """Enhanced scaling and feature selection with multiple approaches."""
    logger.info("Scaling features and selecting important ones...")
    
    # Enhanced scaling approach
    if config.get('use_advanced_scaling', True):
        # Use RobustScaler for most features (handles outliers better)
        scaler = RobustScaler()
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        X_scaled = X.copy()
        X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        
        # Additional StandardScaler for specific feature types that benefit from it
        standard_scaler = StandardScaler()
        rolling_features = [col for col in numeric_columns if 'rolling' in col or 'trend' in col]
        if rolling_features:
            X_scaled[rolling_features] = standard_scaler.fit_transform(X_scaled[rolling_features])
    else:
        scaler = RobustScaler()
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X_scaled = X.copy()
        X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    
    # Fill any remaining NaN values
    X_scaled = X_scaled.fillna(X_scaled.median())
    
    if config['use_feature_selection']:
        logger.info(f"Selecting top {config['k_best_features']} features...")
        try:
            k_best = min(config['k_best_features'], len(X_scaled.columns))
            
            # Use multiple feature selection methods
            # 1. Statistical F-test
            selector_f = SelectKBest(score_func=f_regression, k=k_best)
            
            # 2. Mutual information
            selector_mi = SelectKBest(score_func=mutual_info_regression, k=k_best)
            
            # Fit both selectors
            selector_f.fit(X_scaled, y)
            selector_mi.fit(X_scaled, y)
            
            # Get feature scores
            f_scores = selector_f.scores_
            mi_scores = selector_mi.scores_
            
            # Combine scores (normalize first)
            f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
            mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
            
            combined_scores = 0.6 * f_scores_norm + 0.4 * mi_scores_norm
            
            # Select top features based on combined scores
            top_indices = np.argsort(combined_scores)[-k_best:]
            selected_features = [X_scaled.columns[i] for i in top_indices]
            
            X_selected = X_scaled[selected_features].values
            
            logger.info(f"Selected features: {len(selected_features)} from {len(X_scaled.columns)}")
            
            # Use F-test selector as the main selector for consistency
            selector = selector_f
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            X_selected = X_scaled.values
            selected_features = X_scaled.columns.tolist()
            selector = None
    else:
        X_selected = X_scaled.values
        selected_features = X_scaled.columns.tolist()
        selector = None
    
    return X_selected, selected_features, scaler, selector, numeric_columns, X_scaled.columns.tolist()

def train_models(X_train, y_train, config):
    """Enhanced model training with better hyperparameters and additional models."""
    logger.info("Training enhanced models...")
    
    models = {}
    
    if config['enable_hyperparameter_tuning']:
        logger.info("Performing hyperparameter optimization...")
        models['Random Forest'] = optimize_hyperparameters(X_train, y_train, 'Random Forest', config['cv_folds'])
        models['XGBoost'] = optimize_hyperparameters(X_train, y_train, 'XGBoost', config['cv_folds'])
        models['LightGBM'] = optimize_hyperparameters(X_train, y_train, 'LightGBM', config['cv_folds'])
        
        # Add Gradient Boosting as additional model
        models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    else:
        logger.info("Using enhanced default parameters...")
        models['Random Forest'] = RandomForestRegressor(
            n_estimators=500, 
            max_depth=20, 
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        models['XGBoost'] = XGBRegressor(
            n_estimators=500, 
            max_depth=8, 
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42, 
            n_jobs=-1
        )
        models['LightGBM'] = LGBMRegressor(
            n_estimators=500, 
            max_depth=15,
            learning_rate=0.1,
            num_leaves=50,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            random_state=42, 
            n_jobs=-1,
            verbose=-1
        )
        models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    
    # Train all models
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
    
    # Create enhanced ensemble
    logger.info("Training enhanced ensemble model...")
    
    # Use weighted voting with optimized weights
    weights = config.get('ensemble_weights', [0.25, 0.35, 0.3, 0.1])
    
    ensemble_model = VotingRegressor([
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('lgb', models['LightGBM']),
        ('gb', models['Gradient Boosting'])
    ], weights=weights)
    
    ensemble_model.fit(X_train, y_train)
    models['Gabungan'] = ensemble_model
    
    logger.info("All model training completed!")
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test, y_train_orig, y_test_orig, config, feature_names):
    """Enhanced model evaluation with additional metrics and analysis."""
    logger.info("Evaluating model performance...")
    
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def median_absolute_percentage_error(y_true, y_pred):
        return np.median(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    results = {}
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Make predictions
        y_pred_train = model.predict(X_train_df)
        y_pred_test = model.predict(X_test_df)
        
        # Apply prediction boost factor to counter systematic underestimation
        boost_factor = config.get('prediction_boost_factor', 1.1)
        y_pred_train = y_pred_train * boost_factor
        y_pred_test = y_pred_test * boost_factor
        
        # Reverse transformation if log was used
        if config.get('use_log_transform', False):
            y_pred_train_orig = np.expm1(y_pred_train)
            y_pred_test_orig = np.expm1(y_pred_test)
        else:
            y_pred_train_orig = y_pred_train
            y_pred_test_orig = y_pred_test
        
        # Calculate comprehensive metrics
        results[name] = {
            'train_mae': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'train_rmse': np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig)),
            'train_r2': r2_score(y_train_orig, y_pred_train_orig),
            'test_r2': r2_score(y_test_orig, y_pred_test_orig),
            'train_mape': mean_absolute_percentage_error(y_train_orig, y_pred_train_orig),
            'test_mape': mean_absolute_percentage_error(y_test_orig, y_pred_test_orig),
            'train_medape': median_absolute_percentage_error(y_train_orig, y_pred_train_orig),
            'test_medape': median_absolute_percentage_error(y_test_orig, y_pred_test_orig),
            'mean_prediction_train': np.mean(y_pred_train_orig),
            'mean_prediction_test': np.mean(y_pred_test_orig),
            'mean_actual_train': np.mean(y_train_orig),
            'mean_actual_test': np.mean(y_test_orig)
        }
        
        # Log detailed results
        logger.info(f"{name} Results:")
        logger.info(f"  Test R²: {results[name]['test_r2']:.3f}")
        logger.info(f"  Test MAPE: {results[name]['test_mape']:.2f}%")
        logger.info(f"  Mean Prediction vs Actual (Test): {results[name]['mean_prediction_test']:.2f} vs {results[name]['mean_actual_test']:.2f}")
    
    return results

def save_models_and_artifacts(models, scaler, feature_maps, all_features, selected_features, 
                             numeric_columns, selector, config, results, product_encoder):
    """Enhanced artifact saving with additional metadata."""
    logger.info("Saving models and artifacts...")
    
    # Save all models
    for name, model in models.items():
        joblib.dump(model, MODEL_PATHS[name])
        logger.info(f"Model {name} saved to {MODEL_PATHS[name]}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to {SCALER_PATH}")
    
    # Enhanced artifacts with additional metadata
    artifacts = {
        'feature_maps': feature_maps,
        'all_features_before_selection': all_features,
        'selected_features_after_selection': selected_features,
        'numeric_columns_to_scale': numeric_columns,
        'kbest_selector': selector,
        'config': config,
        'training_results': results,
        'training_timestamp': datetime.now().isoformat(),
        'product_encoder': product_encoder,
        'model_versions': {name: getattr(model, '__version__', 'unknown') 
                          for name, model in models.items()},
        'feature_importance': {}
    }
    
    # Save feature importance for tree-based models
    try:
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(selected_features, model.feature_importances_))
                artifacts['feature_importance'][name] = importance_dict
            elif hasattr(model, 'estimators_'):  # For ensemble models
                if hasattr(model.estimators_[0], 'feature_importances_'):
                    avg_importance = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                    importance_dict = dict(zip(selected_features, avg_importance))
                    artifacts['feature_importance'][name] = importance_dict
    except Exception as e:
        logger.warning(f"Could not save feature importance: {e}")
    
    joblib.dump(artifacts, ARTIFACTS_PATH)
    logger.info(f"Enhanced artifacts saved to {ARTIFACTS_PATH}")

def main(custom_config=None):
    """Enhanced main training pipeline with better error handling and validation."""
    config = CONFIG.copy()
    if custom_config:
        config.update(custom_config)
    
    data_file_path = os.path.join(DATA_DIR, TRAINING_DATA_FILE)
    
    try:
        # Load and preprocess data
        df_agg = load_and_preprocess_data(data_file_path)
        
        if len(df_agg) < 50:  # Increased minimum requirement
            raise ValueError(f"Insufficient data for training. Minimum 50 records required, found {len(df_agg)}")
        
        # Prepare features
        df_enhanced, feature_maps, feature_columns, ohe = prepare_features_and_target(df_agg, config)
        
        # Handle outliers and transform
        X, y_transformed, y_original = handle_outliers_and_transform(df_enhanced, feature_columns, config)
        
        # Enhanced train-test split
        logger.info("Performing time-based train-test split...")
        tscv = TimeSeriesSplit(n_splits=config['cv_folds'])
        train_idx, test_idx = list(tscv.split(X))[-1]
        
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_transformed.iloc[train_idx], y_transformed.iloc[test_idx]
        y_train_orig, y_test_orig = y_original.iloc[train_idx], y_original.iloc[test_idx]
        
        logger.info(f"Training set: {len(X_train_raw)} samples")
        logger.info(f"Test set: {len(X_test_raw)} samples")
        logger.info(f"Training target range: {y_train_orig.min():.2f} - {y_train_orig.max():.2f}")
        logger.info(f"Test target range: {y_test_orig.min():.2f} - {y_test_orig.max():.2f}")
        
        # Scale and select features
        X_train, selected_features, scaler, selector, numeric_columns, all_feature_names = scale_and_select_features(
            X_train_raw.copy(), y_train.copy(), config
        )
        
        # Prepare test set
        X_test_scaled = X_test_raw.copy()
        X_test_scaled[numeric_columns] = scaler.transform(X_test_raw[numeric_columns])
        X_test_scaled = X_test_scaled.fillna(X_train_raw.median())
        
        if config['use_feature_selection'] and selector is not None:
            X_test = selector.transform(X_test_scaled[all_feature_names])
        else:
            X_test = X_test_scaled[selected_features].values

        # Train models
        models = train_models(X_train, y_train, config)
        
        # Evaluate models
        results = evaluate_models(models, X_train, X_test, y_train, y_test, 
                                y_train_orig, y_test_orig, config, selected_features)
        
        # Save everything
        save_models_and_artifacts(models, scaler, feature_maps, all_feature_names, 
                                selected_features, numeric_columns, selector, config, results, ohe)
        
        # Print summary
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info("\nModel Performance Summary:")
        for name, metrics in results.items():
            logger.info(f"{name:15s} | R²: {metrics['test_r2']:.3f} | MAPE: {metrics['test_mape']:.1f}% | Pred Mean: {metrics['mean_prediction_test']:.1f}")
        
        # Find best model
        best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
        logger.info(f"\nBest performing model: {best_model} (R² = {results[best_model]['test_r2']:.3f})")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        main()
        print("\nTraining completed successfully!")
        print("Check 'training.log' for detailed results")
    except Exception as e:
        print(f"\nTraining failed. Check 'training.log' for details. Error: {e}")
        exit(1)