import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, scaler=None):
        self.scaler = scaler
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various evaluation metrics."""
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': self._calculate_mape(y_true, y_pred)
        }
        
        logger.info("Evaluation metrics calculated")
        return metrics
        
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Calculate MAPE in chunks to avoid memory issues
        chunk_size = 10000  # Process 10000 elements at a time
        total_mape = 0
        count = 0
        
        for i in range(0, len(y_true), chunk_size):
            chunk_true = y_true[i:i + chunk_size]
            chunk_pred = y_pred[i:i + chunk_size]
            
            # Only consider non-zero values
            mask = chunk_true != 0
            if np.any(mask):
                chunk_mape = np.abs((chunk_true[mask] - chunk_pred[mask]) / chunk_true[mask])
                total_mape += np.sum(chunk_mape)
                count += np.sum(mask)
        
        return (total_mape / count) * 100 if count > 0 else 0
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Actual vs Predicted Values") -> None:
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        # Plot every 100th point to avoid memory issues
        plt.plot(y_true[::100], label='Actual', alpha=0.7)
        plt.plot(y_pred[::100], label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Power Consumption')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot residuals to check for patterns."""
        # Calculate residuals in chunks
        chunk_size = 10000
        residuals = []
        predictions = []
        
        for i in range(0, len(y_true), chunk_size):
            chunk_true = y_true[i:i + chunk_size]
            chunk_pred = y_pred[i:i + chunk_size]
            chunk_residuals = chunk_true - chunk_pred
            residuals.extend(chunk_residuals)
            predictions.extend(chunk_pred)
        
        # Convert to numpy arrays
        residuals = np.array(residuals)
        predictions = np.array(predictions)
        
        plt.figure(figsize=(12, 6))
        # Plot every 100th point to avoid memory issues
        plt.scatter(predictions[::100], residuals[::100], alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.show()
        
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot distribution of prediction errors."""
        # Calculate errors in chunks
        chunk_size = 10000
        errors = []
        
        for i in range(0, len(y_true), chunk_size):
            chunk_true = y_true[i:i + chunk_size]
            chunk_pred = y_pred[i:i + chunk_size]
            chunk_errors = chunk_true - chunk_pred
            errors.extend(chunk_errors)
        
        # Convert to numpy array
        errors = np.array(errors)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
    def plot_feature_importance(self, model, feature_names: list) -> None:
        """Plot feature importance if available."""
        try:
            # For models that support feature importance
            importance = model.feature_importances_
            plt.figure(figsize=(12, 6))
            sns.barplot(x=importance, y=feature_names)
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.grid(True)
            plt.show()
        except:
            logger.warning("Feature importance not available for this model type")
            
    def create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               feature_names: list = None) -> pd.DataFrame:
        """Create a comprehensive evaluation report."""
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create time series analysis
        residuals = y_true - y_pred
        error_stats = {
            'Mean Error': np.mean(residuals),
            'Std Error': np.std(residuals),
            'Min Error': np.min(residuals),
            'Max Error': np.max(residuals)
        }
        
        # Combine all metrics
        report = {
            **metrics,
            **error_stats
        }
        
        return pd.DataFrame([report])
        
    def plot_learning_curve(self, history) -> None:
        """Plot the learning curve from training history."""
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show() 