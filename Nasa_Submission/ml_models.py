"""
Machine Learning models for pollutant concentration forecasting.
Includes Random Forest, Gradient Boosting, and LSTM models.
"""

from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Conditional imports with fallbacks
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Using simplified implementations.")
    SKLEARN_AVAILABLE = False
    # Fallback numpy-like functions
    class np:
        @staticmethod
        def array(x): return x
        @staticmethod
        def nan_to_num(x): return x
        @staticmethod
        def sqrt(x): return x**0.5
        nan = None
    # Fallback pandas-like class
    class pd:
        class DatetimeIndex: pass
        @staticmethod
        def isna(x): return x is None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM models disabled.")
    TENSORFLOW_AVAILABLE = False
    # Fallback TensorFlow classes
    class keras:
        class Sequential: pass
        class models:
            @staticmethod
            def load_model(path): pass
    class layers: pass

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    print("Warning: joblib not available. Model saving disabled.")
    JOBLIB_AVAILABLE = False
    class joblib:
        @staticmethod
        def dump(obj, path): pass
        @staticmethod
        def load(path): return {}


class PollutantPredictor:
    """
    Multi-model predictor for pollutant concentrations.
    
    Supports Random Forest, Gradient Boosting, and LSTM models
    for forecasting multiple pollutants simultaneously.
    """
    
    def __init__(self, model_type: str = 'random_forest', **model_params):
        """
        Initialize predictor with specified model type.
        
        Args:
            model_type: 'random_forest', 'gradient_boosting', or 'lstm'
            **model_params: Additional parameters for the model
        """
        if not SKLEARN_AVAILABLE and model_type in ['random_forest', 'gradient_boosting']:
            raise ImportError(f"scikit-learn required for {model_type} model. Install with: pip install scikit-learn")
        
        if not TENSORFLOW_AVAILABLE and model_type == 'lstm':
            raise ImportError("TensorFlow required for LSTM model. Install with: pip install tensorflow")
        
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        
        if SKLEARN_AVAILABLE:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        else:
            self.scaler_X = None
            self.scaler_y = None
            
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        
        # Default parameters for each model type
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'recurrent_dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'sequence_length': 24  # 24 hours lookback
            }
        }
        
        # Merge default parameters with provided ones
        if model_type in self.default_params:
            self.params = {**self.default_params[model_type], **model_params}
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_model(self, n_features: int, n_targets: int):
        """Create model based on specified type."""
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(**{k: v for k, v in self.params.items() 
                                                if k != 'sequence_length'})
            self.model = MultiOutputRegressor(base_model)
            
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(**{k: v for k, v in self.params.items() 
                                                    if k != 'sequence_length'})
            self.model = MultiOutputRegressor(base_model)
            
        elif self.model_type == 'lstm':
            self.model = self._create_lstm_model(n_features, n_targets)
    
    def _create_lstm_model(self, n_features: int, n_targets: int):
        """Create LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.params['units'], 
                       return_sequences=True,
                       dropout=self.params['dropout'],
                       recurrent_dropout=self.params['recurrent_dropout'],
                       input_shape=(self.params['sequence_length'], n_features)),
            
            layers.LSTM(self.params['units'] // 2,
                       dropout=self.params['dropout'],
                       recurrent_dropout=self.params['recurrent_dropout']),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(n_targets)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def _prepare_lstm_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """Prepare sequences for LSTM training/prediction."""
        seq_length = self.params['sequence_length']
        
        if len(X) < seq_length:
            raise ValueError(f"Need at least {seq_length} samples for LSTM")
        
        # Create sequences
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame, validation_split: float = 0.2):
        """
        Train the model on the provided data.
        
        Args:
            X: Feature DataFrame
            y: Target DataFrame (pollutant concentrations)
            validation_split: Fraction of data to use for validation
        """
        # Store feature and target names
        self.feature_names = list(X.columns)
        self.target_names = list(y.columns)
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32)
        
        # Handle missing values
        X_array = np.nan_to_num(X_array)
        y_array = np.nan_to_num(y_array)
        
        # Scale features and targets
        X_scaled = self.scaler_X.fit_transform(X_array)
        y_scaled = self.scaler_y.fit_transform(y_array)
        
        # Create model
        self._create_model(X_scaled.shape[1], y_scaled.shape[1])
        
        if self.model_type == 'lstm':
            # Prepare sequences for LSTM
            X_seq, y_seq = self._prepare_lstm_sequences(X_scaled, y_scaled)
            
            # Split data for LSTM (time series split)
            split_idx = int(len(X_seq) * (1 - validation_split))
            X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
            y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
            
            # Train LSTM
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.params['epochs'],
                batch_size=self.params['batch_size'],
                verbose=0
            )
            self.training_history = history.history
            
        else:
            # For scikit-learn models, use regular train-test split
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, 
                test_size=validation_split, 
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Calculate validation metrics
        if self.model_type == 'lstm':
            val_pred = self.model.predict(X_val)
        else:
            val_pred = self.model.predict(X_val)
        
        self.validation_metrics = self._calculate_metrics(y_val, val_pred)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted pollutant concentrations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Ensure feature consistency
        if list(X.columns) != self.feature_names:
            raise ValueError("Feature columns don't match training data")
        
        # Preprocess
        X_array = X.values.astype(np.float32)
        X_array = np.nan_to_num(X_array)
        X_scaled = self.scaler_X.transform(X_array)
        
        if self.model_type == 'lstm':
            # For LSTM, we need sequences
            if len(X_scaled) < self.params['sequence_length']:
                raise ValueError(f"Need at least {self.params['sequence_length']} samples for LSTM prediction")
            
            X_seq = self._prepare_lstm_sequences(X_scaled)
            predictions_scaled = self.model.predict(X_seq)
        else:
            predictions_scaled = self.model.predict(X_scaled)
        
        # Inverse transform predictions
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def predict_next_hours(self, X: pd.DataFrame, n_hours: int = 24) -> pd.DataFrame:
        """
        Predict pollutant concentrations for the next n hours.
        
        Args:
            X: Historical feature data
            n_hours: Number of hours to predict
            
        Returns:
            DataFrame with predicted concentrations
        """
        predictions = []
        current_X = X.copy()
        
        for hour in range(n_hours):
            # Make prediction for next hour
            tail_size = self.params.get('sequence_length', 1)
            tail_data = current_X.tail(tail_size)
            pred = self.predict(tail_data)
            
            if pred.ndim == 2:
                pred = pred[-1]  # Take the last prediction if multiple
            
            predictions.append(pred)
            
            # For iterative prediction, we would update current_X with prediction
            # This is simplified - in practice, you'd need to update features properly
            
        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions, columns=self.target_names)
        
        # Add time index
        if isinstance(X.index, pd.DatetimeIndex):
            start_time = X.index[-1] + pd.Timedelta(hours=1)
            pred_df.index = pd.date_range(start_time, periods=n_hours, freq='H')
        
        return pred_df
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        # Inverse transform for metric calculation
        y_true_orig = self.scaler_y.inverse_transform(y_true)
        y_pred_orig = self.scaler_y.inverse_transform(y_pred)
        
        metrics = {}
        for i, target in enumerate(self.target_names):
            metrics[target] = {
                'mae': mean_absolute_error(y_true_orig[:, i], y_pred_orig[:, i]),
                'rmse': np.sqrt(mean_squared_error(y_true_orig[:, i], y_pred_orig[:, i])),
                'r2': r2_score(y_true_orig[:, i], y_pred_orig[:, i])
            }
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Handle LSTM sequence offset
        if self.model_type == 'lstm':
            seq_length = self.params['sequence_length']
            y_test_aligned = y_test.iloc[seq_length:].values
        else:
            y_test_aligned = y_test.values
        
        # Calculate metrics
        y_test_scaled = self.scaler_y.transform(y_test_aligned)
        pred_scaled = self.scaler_y.transform(predictions)
        
        return self._calculate_metrics(y_test_scaled, pred_scaled)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'model_type': self.model_type,
            'params': self.params,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'validation_metrics': getattr(self, 'validation_metrics', None)
        }
        
        if self.model_type == 'lstm':
            # Save Keras model separately
            self.model.save(f"{filepath}_lstm_model")
            model_data['model_path'] = f"{filepath}_lstm_model"
        else:
            model_data['model'] = self.model
        
        joblib.dump(model_data, f"{filepath}.pkl")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a saved model."""
        model_data = joblib.load(f"{filepath}.pkl")
        
        # Create instance
        instance = cls(model_data['model_type'], **model_data['params'])
        
        # Restore attributes
        instance.scaler_X = model_data['scaler_X']
        instance.scaler_y = model_data['scaler_y']
        instance.feature_names = model_data['feature_names']
        instance.target_names = model_data['target_names']
        instance.validation_metrics = model_data.get('validation_metrics')
        
        # Load model
        if model_data['model_type'] == 'lstm':
            instance.model = keras.models.load_model(model_data['model_path'])
        else:
            instance.model = model_data['model']
        
        instance.is_fitted = True
        return instance


def train_multiple_models(X: pd.DataFrame, y: pd.DataFrame, 
                         test_size: float = 0.2) -> Dict[str, PollutantPredictor]:
    """
    Train and compare multiple model types.
    
    Args:
        X: Features DataFrame
        y: Targets DataFrame
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary of trained models
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    models = {}
    model_types = ['random_forest', 'gradient_boosting', 'lstm']
    
    for model_type in model_types:
        print(f"Training {model_type} model...")
        
        try:
            model = PollutantPredictor(model_type)
            model.fit(X_train, y_train)
            
            # Evaluate
            test_metrics = model.evaluate(X_test, y_test)
            model.test_metrics = test_metrics
            
            models[model_type] = model
            print(f"{model_type} training completed")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
    
    return models


if __name__ == "__main__":
    print("Pollutant Prediction Models Module")
    print("Available models: Random Forest, Gradient Boosting, LSTM")
    print("Use PollutantPredictor class to train and make predictions")