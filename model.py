import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer, KNNImputer
import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class AdvancedSolarPanelPredictor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.model = None
        self.feature_columns = None
        self.imputer = None
        self.power_transformer = None
        self.kmeans = None

    def advanced_feature_engineering(self, df, is_training=True):
        """Advanced feature engineering with domain knowledge"""
        df = df.copy()

        # Convert object columns that should be numeric
        numeric_cols = ['humidity', 'wind_speed', 'pressure']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # ===== BASIC ENGINEERED FEATURES =====
        if 'voltage' in df.columns and 'current' in df.columns:
            df['power'] = df['voltage'] * df['current']
            df['power_density'] = df['power'] / (df['irradiance'] + 1e-6)

        if 'temperature' in df.columns and 'module_temperature' in df.columns:
            df['temp_diff'] = df['module_temperature'] - df['temperature']
            df['temp_ratio'] = df['module_temperature'] / (df['temperature'] + 273.15)

        if 'irradiance' in df.columns and 'cloud_coverage' in df.columns:
            df['effective_irradiance'] = df['irradiance'] * (1 - df['cloud_coverage']/100)
            df['irradiance_cloud_interaction'] = df['irradiance'] * df['cloud_coverage']

        # ===== ADVANCED SOLAR PHYSICS FEATURES =====
        if 'panel_age' in df.columns:
            df['age_squared'] = df['panel_age'] ** 2
            df['age_cubed'] = df['panel_age'] ** 3
            # Degradation rate typically 0.5-0.8% per year
            df['expected_degradation'] = 1 - (0.006 * df['panel_age'])

        if 'soiling_ratio' in df.columns:
            df['cleanliness'] = 1 - df['soiling_ratio']
            df['soiling_squared'] = df['soiling_ratio'] ** 2

        # ===== MAINTENANCE EFFECTIVENESS =====
        if 'maintenance_count' in df.columns and 'panel_age' in df.columns:
            df['maintenance_per_year'] = df['maintenance_count'] / (df['panel_age'] + 1)
            df['maintenance_recency'] = df['panel_age'] - df['maintenance_count']
            df['over_maintained'] = (df['maintenance_count'] > df['panel_age'] * 2).astype(int)

        # ===== ENVIRONMENTAL INTERACTIONS =====
        if 'wind_speed' in df.columns and 'module_temperature' in df.columns:
            df['cooling_effect'] = df['wind_speed'] / (df['module_temperature'] + 1)

        if 'humidity' in df.columns and 'temperature' in df.columns:
            df['heat_index'] = df['temperature'] + 0.5 * df['humidity']

        if 'pressure' in df.columns:
            df['pressure_normalized'] = (df['pressure'] - 1013.25) / 1013.25

        # ===== PERFORMANCE RATIOS =====
        if 'irradiance' in df.columns and 'voltage' in df.columns:
            df['voltage_per_irradiance'] = df['voltage'] / (df['irradiance'] + 1e-6)

        if 'current' in df.columns and 'irradiance' in df.columns:
            df['current_per_irradiance'] = df['current'] / (df['irradiance'] + 1e-6)

        # ===== CATEGORICAL INTERACTIONS =====
        if 'installation_type' in df.columns and 'irradiance' in df.columns:
            # Create interaction between installation type and irradiance
            df['installation_irradiance_interaction'] = df['irradiance'] * pd.get_dummies(df['installation_type'], prefix='install').sum(axis=1)

        # ===== POLYNOMIAL FEATURES FOR KEY VARIABLES =====
        key_vars = ['irradiance', 'temperature', 'voltage', 'current']
        for var in key_vars:
            if var in df.columns:
                df[f'{var}_squared'] = df[var] ** 2
                df[f'{var}_sqrt'] = np.sqrt(np.abs(df[var]))

        # ===== CLUSTERING FEATURES =====
        if is_training and 'irradiance' in df.columns and 'temperature' in df.columns:
            # Create weather condition clusters
            weather_features = ['irradiance', 'temperature', 'humidity', 'cloud_coverage', 'wind_speed']
            weather_data = df[[col for col in weather_features if col in df.columns]].fillna(0)

            self.kmeans = KMeans(n_clusters=8, random_state=42)
            df['weather_cluster'] = self.kmeans.fit_predict(weather_data)
        elif hasattr(self, 'kmeans') and self.kmeans is not None:
            weather_features = ['irradiance', 'temperature', 'humidity', 'cloud_coverage', 'wind_speed']
            weather_data = df[[col for col in weather_features if col in df.columns]].fillna(0)
            df['weather_cluster'] = self.kmeans.predict(weather_data)

        return df

    def encode_categorical_features(self, df, is_training=True):
        """Enhanced categorical encoding with frequency encoding"""
        df = df.copy()
        categorical_cols = ['string_id', 'error_code', 'installation_type']

        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    df[col] = df[col].fillna('Unknown')

                    # Frequency encoding for high-cardinality features
                    if col == 'string_id':
                        freq_map = df[col].value_counts().to_dict()
                        df[f'{col}_frequency'] = df[col].map(freq_map)
                        self.label_encoders[f'{col}_frequency'] = freq_map

                    # Label encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    df[col] = df[col].fillna('Unknown')

                    # Apply frequency encoding
                    if f'{col}_frequency' in self.label_encoders:
                        freq_map = self.label_encoders[f'{col}_frequency']
                        df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)

                    # Apply label encoding
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        unique_vals = df[col].unique()
                        for val in unique_vals:
                            if val not in le.classes_:
                                le.classes_ = np.append(le.classes_, val)
                        df[col] = le.transform(df[col])
                    else:
                        df[col] = 0

        return df

    def get_feature_columns(self, df):
        """Get the list of feature columns (excluding id and target)"""
        exclude_cols = {'id', 'efficiency'}
        return [col for col in df.columns if col not in exclude_cols]

    def prepare_features(self, df, is_training=True):
        """Complete feature preparation pipeline with advanced preprocessing"""
        # Step 1: Advanced feature engineering
        df_processed = self.advanced_feature_engineering(df, is_training)

        # Step 2: Handle categorical variables
        df_processed = self.encode_categorical_features(df_processed, is_training)

        # Step 3: Get feature columns
        if is_training:
            self.feature_columns = self.get_feature_columns(df_processed)
            print(f"Feature columns defined: {len(self.feature_columns)} features")

        # Step 4: Select only the feature columns
        available_features = [col for col in self.feature_columns if col in df_processed.columns]
        X = df_processed[available_features].copy()

        # Step 5: Handle missing features for test data
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        # Ensure correct order
        X = X[self.feature_columns]

        # Step 6: Handle missing values with advanced imputation
        if is_training:
            # Use iterative imputer for better missing value handling
            self.imputer = KNNImputer(n_neighbors=7, weights='distance')
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = pd.DataFrame(
                self.imputer.transform(X),
                columns=X.columns,
                index=X.index
            )

        # Step 7: Apply power transformation for normality
        if is_training:
            self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            X_transformed = pd.DataFrame(
                self.power_transformer.fit_transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )
        else:
            X_transformed = pd.DataFrame(
                self.power_transformer.transform(X_imputed),
                columns=X_imputed.columns,
                index=X_imputed.index
            )

        # Step 8: Scale features
        if is_training:
            self.scaler = RobustScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_transformed),
                columns=X_transformed.columns,
                index=X_transformed.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_transformed),
                columns=X_transformed.columns,
                index=X_transformed.index
            )

        return X_scaled

    def create_advanced_ensemble(self):
        """Create an advanced ensemble with optimized hyperparameters"""
        models = {
            'lgb1': lgb.LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=12,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_samples=20,
                random_state=42,
                verbose=-1
            ),
            'lgb2': lgb.LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=10,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_samples=30,
                random_state=123,
                verbose=-1
            ),
            'xgb1': xgb.XGBRegressor(
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_weight=3,
                random_state=42,
                verbosity=0
            ),
            'xgb2': xgb.XGBRegressor(
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.9,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=0.2,
                min_child_weight=5,
                random_state=123,
                verbosity=0
            ),
            'rf': RandomForestRegressor(
                n_estimators=800,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=800,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=1000,
                learning_rate=0.02,
                max_depth=10,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            )
        }

        return WeightedVotingEnsemble(models)

    def train(self, train_df):
        """Train the advanced model with cross-validation"""
        print("Preprocessing and preparing training features...")
        X = self.prepare_features(train_df, is_training=True)
        y = train_df['efficiency'].copy()

        print(f"Training features shape: {X.shape}")
        print(f"Training feature columns: {list(X.columns)[:10]}...")
        print(f"Target shape: {y.shape}")

        print("Training advanced ensemble...")
        self.model = self.create_advanced_ensemble()
        self.model.fit(X, y)

        # Evaluate on training data
        train_pred = self.model.predict(X)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred))
        train_r2 = r2_score(y, train_pred)

        print(f"Training RMSE: {train_rmse:.6f}")
        print(f"Training R²: {train_r2:.6f}")
        print(f"Training Score: {100*(1-train_rmse):.2f}")

        return self

    def predict(self, test_df):
        """Make predictions on test data"""
        print("Preprocessing and preparing test features...")
        X_test = self.prepare_features(test_df, is_training=False)

        print(f"Test features shape: {X_test.shape}")
        print(f"Test feature columns: {list(X_test.columns)[:10]}...")

        print("Making predictions...")
        predictions = self.model.predict(X_test)

        # Ensure predictions are within reasonable bounds
        predictions = np.clip(predictions, 0, 1)

        return predictions

class WeightedVotingEnsemble(BaseEstimator, RegressorMixin):
    """Advanced weighted voting ensemble with cross-validation optimization"""
    def __init__(self, models):
        self.models = models
        self.weights = None

    def fit(self, X, y):
        # Use stratified k-fold for better validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Train all models with cross-validation
        cv_scores = {}

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)

            # Cross-validation scoring
            scores = cross_val_score(model, X, y, cv=kf,
                                   scoring='neg_mean_squared_error', n_jobs=-1)
            rmse_score = np.sqrt(-scores.mean())
            cv_scores[name] = rmse_score
            print(f"{name} CV RMSE: {rmse_score:.6f}")

        # Calculate optimal weights
        self.weights = {}
        total_weight = 0

        # Use inverse RMSE for weighting (lower RMSE = higher weight)
        for name, rmse in cv_scores.items():
            weight = 1 / (rmse ** 2)  # Square for more emphasis on best models
            self.weights[name] = weight
            total_weight += weight

        # Normalize weights
        for name in self.weights:
            self.weights[name] /= total_weight
            print(f"{name} Final Weight: {self.weights[name]:.4f}")

        return self

    def predict(self, X):
        predictions = np.zeros(len(X))

        for name, model in self.models.items():
            pred = model.predict(X)
            predictions += self.weights[name] * pred

        return predictions

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Initialize and train the advanced predictor
    predictor = AdvancedSolarPanelPredictor()
    predictor.train(train_df)

    # Make predictions
    predictions = predictor.predict(test_df)

    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'efficiency': predictions
    })

    print(f"\nPredictions statistics:")
    print(f"Mean: {predictions.mean():.6f}")
    print(f"Std: {predictions.std():.6f}")
    print(f"Min: {predictions.min():.6f}")
    print(f"Max: {predictions.max():.6f}")

    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission saved to 'submission.csv'")
    print(f"Submission shape: {submission.shape}")

    return predictor, submission

if __name__ == "__main__":
    predictor, submission = main()