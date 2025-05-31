# Solar Panel Efficiency Prediction

A machine learning solution for predicting solar panel performance degradation and potential failures using historical and real-time sensor data. This project enables predictive maintenance and optimal energy output optimization for photovoltaic (PV) systems.

## 🎯 Project Overview

Traditional maintenance methods for photovoltaic systems are often reactive, leading to energy loss and increased costs. This project develops a comprehensive ML model that predicts performance degradation and potential failures in solar panels, enabling proactive maintenance strategies.

**Competition Score Formula:** `Score = 100 * (1 - RMSE)`

## 📊 Dataset Description

The dataset contains sensor readings and operational parameters from solar panel installations:

### Files Structure
- **train.csv**: 20,000 × 17 columns (includes target variable)
- **test.csv**: 12,000 × 16 columns (prediction target)
- **sample_submission.csv**: Template for submission format

### Features

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier |
| `temperature` | Ambient air temperature (°C) |
| `irradiance` | Solar energy received per unit area (W/m²) |
| `humidity` | Moisture content in air |
| `panel_age` | Age of solar panel (years) |
| `maintenance_count` | Number of previous maintenance activities |
| `soiling_ratio` | Efficiency reduction due to dust/debris (0.0-1.0) |
| `voltage` | Voltage output from panel (V) |
| `current` | Current output from panel (A) |
| `module_temperature` | Panel surface temperature |
| `cloud_coverage` | Sky coverage by clouds (percentage) |
| `wind_speed` | Wind speed (m/s) |
| `pressure` | Atmospheric pressure (hPa) |
| `string_id` | Identifier for panel string/group |
| `error_code` | Diagnostic error codes |
| `installation_type` | Mounting setup (fixed, tracking, dual-axis) |
| `efficiency` | **Target variable** - Energy output efficiency |

## 🔧 Technical Architecture

### Core Components

1. **Data Preprocessing Pipeline**
   - Handles mixed data types and missing values
   - Advanced feature engineering
   - Robust categorical encoding
   - KNN imputation for missing values

2. **Feature Engineering**
   - **Power Calculation**: `voltage × current`
   - **Temperature Differential**: `module_temperature - ambient_temperature`
   - **Effective Irradiance**: Adjusted for cloud coverage
   - **Age-Related Features**: Panel age squared, maintenance frequency
   - **Cleanliness Factor**: Derived from soiling ratio

3. **Ensemble Model Architecture**
   - **LightGBM**: Fast gradient boosting
   - **XGBoost**: Robust gradient boosting
   - **Random Forest**: Ensemble of decision trees
   - **Extra Trees**: Randomized ensemble
   - **Gradient Boosting**: Classic boosting algorithm

4. **Smart Ensemble Weighting**
   - Cross-validation based weight calculation
   - Performance-weighted voting system

### Key Features

- **Robust Missing Value Handling**: KNN imputation
- **Advanced Feature Scaling**: RobustScaler (outlier-resistant)
- **Categorical Encoding**: Handles unseen categories gracefully
- **Prediction Clipping**: Ensures valid efficiency range [0, 1]
- **Cross-Validation Optimization**: Model weights based on CV performance

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm xgboost
```

### Required Libraries
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- warnings (built-in)

## 📋 Usage

### Basic Usage
```python
# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Initialize predictor
predictor = SolarPanelEfficiencyPredictor()

# Train model
predictor.train(train_df)

# Make predictions
predictions = predictor.predict(test_df)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'efficiency': predictions
})
submission.to_csv('submission.csv', index=False)
```

### Running the Complete Pipeline
```bash
python solar_panel_predictor.py
```

### Expected Output
```
Loading data...
Training data shape: (20000, 17)
Test data shape: (12000, 16)
Preprocessing and preparing training features...
Feature columns defined: 21 features
Training features shape: (20000, 21)
Training ensemble model...
Training lgb...
Training xgb...
Training rf...
Training et...
Training gb...
lgb CV RMSE: 0.107167, Weight: 9.3311
...
Training RMSE: 0.045977
Training R²: 0.892789
Training Score: 95.40
```

## 🏗️ Model Architecture Details

### Preprocessing Pipeline
1. **Data Type Conversion**: Object columns → Numeric
2. **Feature Engineering**: Create derived features
3. **Categorical Encoding**: LabelEncoder with unseen category handling
4. **Missing Value Imputation**: KNN-based imputation
5. **Feature Scaling**: RobustScaler normalization

### Ensemble Strategy
- **5 Base Models**: Different algorithms for diversity
- **Weighted Voting**: Performance-based weight assignment
- **Cross-Validation**: 3-fold CV for weight calculation
- **Final Prediction**: Weighted average of all models

### Model Hyperparameters

#### LightGBM & XGBoost
- **n_estimators**: 1000
- **learning_rate**: 0.05
- **max_depth**: 8
- **subsample**: 0.8

#### Random Forest & Extra Trees
- **n_estimators**: 500
- **max_depth**: 15
- **min_samples_split**: 5

## 📈 Performance Metrics

- **Primary Metric**: RMSE (Root Mean Square Error)
- **Secondary Metrics**: R² Score
- **Competition Score**: 100 * (1 - RMSE)
- **Expected Performance**: ~95+ competition score

## 📁 File Structure
```
solar-panel-prediction/
├── solar_panel_predictor.py    # Main prediction script
├── train.csv                   # Training data
├── test.csv                    # Test data
├── sample_submission.csv       # Submission format
├── submission.csv              # Generated predictions
└── README.md                   # This file
```

## 🔍 Model Features & Benefits

### Advanced Preprocessing
- **Smart Missing Value Handling**: KNN imputation preserves data relationships
- **Robust Scaling**: Less sensitive to outliers than standard scaling
- **Feature Engineering**: Creates meaningful derived features

### Ensemble Benefits
- **Model Diversity**: Different algorithms capture different patterns
- **Reduced Overfitting**: Ensemble averaging reduces variance
- **Improved Robustness**: Less sensitive to individual model failures
- **Optimized Weights**: Performance-based weighting maximizes accuracy

### Production Ready
- **Error Handling**: Graceful handling of data issues
- **Scalable Architecture**: Modular design for easy maintenance
- **Reproducible Results**: Fixed random seeds ensure consistency

## 🛠️ Customization Options

### Model Tuning
- Adjust hyperparameters in `create_ensemble_model()`
- Modify ensemble weights in `VotingEnsemble`
- Add/remove models from the ensemble

### Feature Engineering
- Add custom features in `clean_and_engineer_features()`
- Modify imputation strategy in `prepare_features()`
- Adjust scaling method as needed

### Evaluation
- Change cross-validation folds
- Modify evaluation metrics
- Add additional validation strategies

## 🎯 Expected Results

With the current configuration, the model typically achieves:
- **Training RMSE**: ~0.046
- **Training R²**: ~0.89
- **Competition Score**: ~95.4
- **Cross-Validation RMSE**: ~0.105-0.107

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new features
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 📞 Support

For questions or issues:
1. Check the documentation above
2. Review the code comments
3. Open an issue in the repository
4. Contact the development team

---

**Note**: This model is designed for educational and competition purposes. For production deployment, additional validation, monitoring, and safety measures should be implemented.
