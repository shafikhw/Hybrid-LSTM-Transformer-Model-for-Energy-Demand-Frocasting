import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from evaluator import ModelEvaluator
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU
def configure_gpu():
    # List all physical devices
    physical_devices = tf.config.list_physical_devices()
    logger.info(f"Available physical devices: {physical_devices}")
    
    # List GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"Available GPUs: {gpus}")
    
    if gpus:
        try:
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth enabled")
            
            # Set visible devices
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logger.info(f"Using GPU: {gpus[0]}")
            
            # Verify GPU is being used
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                logger.info("GPU test operation successful")
                
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
    else:
        logger.warning("No GPU available. Running on CPU.")

def load_model_and_predict():
    # Configure GPU
    configure_gpu()
    
    # Initialize components
    data_processor = DataProcessor('household_power_consumption.txt')
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    data = data_processor.load_and_preprocess()
    scaled_data = data_processor.scale_data()
    
    # Create sequences (using same sequence length as training)
    seq_length = 24
    X, y = data_processor.create_sequences(seq_length)
    
    # Split data (using same split as training)
    data_splits = data_processor.split_data(X, y)
    X_test, y_test = data_splits['test']
    
    # Load the trained model
    logger.info("Loading trained model...")
    try:
        # Set device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Load model with GPU
        with tf.device('/GPU:0'):
            model = tf.keras.models.load_model('lstm_best.h5')
            logger.info("Model loaded successfully on GPU")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return
    
    # Make predictions
    logger.info("Making predictions...")
    with tf.device('/GPU:0'):
        y_pred = model.predict(X_test, batch_size=32)
    
    # Inverse transform predictions and actual values
    scaler = data_processor.scaler
    y_test_original = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), scaled_data.shape[1]-1))]))[:, 0]
    y_pred_original = scaler.inverse_transform(np.column_stack([y_pred, np.zeros((len(y_pred), scaled_data.shape[1]-1))]))[:, 0]
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test_original, y_pred_original)
    logger.info("\nPrediction Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.plot(y_test_original[:100], label='Actual', alpha=0.7)
    plt.plot(y_pred_original[:100], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted (First 100 samples)')
    plt.xlabel('Time')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Residuals
    plt.subplot(2, 2, 2)
    residuals = y_test_original - y_pred_original
    plt.scatter(y_pred_original, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Plot 3: Error Distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot 4: Cumulative Distribution
    plt.subplot(2, 2, 4)
    sorted_errors = np.sort(np.abs(residuals))
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf)
    plt.title('Cumulative Error Distribution')
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some sample predictions
    logger.info("\nSample Predictions:")
    for i in range(5):
        logger.info(f"Actual: {y_test_original[i]:.4f}, Predicted: {y_pred_original[i]:.4f}, Error: {residuals[i]:.4f}")
    
    # Save predictions to CSV
    results = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': y_pred_original,
        'Error': residuals
    })
    results.to_csv('prediction_results.csv', index=False)
    logger.info("\nPredictions saved to 'prediction_results.csv'")

if __name__ == "__main__":
    load_model_and_predict() 