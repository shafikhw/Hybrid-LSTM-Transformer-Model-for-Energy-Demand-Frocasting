import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess the data with improved handling."""
        try:
            # Load data with proper datetime parsing
            data = pd.read_csv(
                self.file_path,
                sep=";",
                parse_dates=[['Date', 'Time']],
                dayfirst=True,
                low_memory=False
            )
            
            # Convert to datetime and set as index
            data['Date_Time'] = pd.to_datetime(data['Date_Time'])
            data.set_index('Date_Time', inplace=True)
            data.sort_index(inplace=True)
            
            # Handle missing values and convert to numeric
            data.replace('?', np.nan, inplace=True)
            data = data.apply(pd.to_numeric)
            
            # Advanced missing value handling
            self._handle_missing_values(data)
            
            # Add time-based features
            self._add_time_features(data)
            
            self.data = data
            logger.info("Data loaded and preprocessed successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _handle_missing_values(self, data: pd.DataFrame) -> None:
        """Handle missing values using advanced techniques."""
        # Forward fill for short gaps
        data.ffill(limit=3, inplace=True)
        
        # Backward fill for remaining gaps
        data.bfill(limit=3, inplace=True)
        
        # For larger gaps, use interpolation
        for col in data.columns:
            if data[col].isna().sum() > 0:
                data[col] = data[col].interpolate(method='time')
                
    def _add_time_features(self, data: pd.DataFrame) -> None:
        """Add time-based features to improve model performance."""
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
        
        # Add cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
        data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
        
    def scale_data(self) -> pd.DataFrame:
        """Scale the data using MinMaxScaler."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_and_preprocess first.")
            
        self.scaled_data = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            columns=self.data.columns,
            index=self.data.index
        )
        return self.scaled_data
        
    def create_sequences(self, seq_length: int, target_col: str = 'Global_active_power') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        if self.scaled_data is None:
            raise ValueError("Data not scaled. Call scale_data first.")
            
        X, y = [], []
        data = self.scaled_data.values
        
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length, self.scaled_data.columns.get_loc(target_col)])
            
        # Convert to float32 to ensure consistency
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets."""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        } 