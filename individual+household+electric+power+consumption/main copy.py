import logging
from data_processor_copy import DataProcessor
from model_copy import PowerConsumptionModel
from evaluator import ModelEvaluator
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which is more reliable on Windows
import matplotlib.pyplot as plt
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    data_processor = DataProcessor('household_power_consumption.txt')
    evaluator = ModelEvaluator()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    start_time = time.time()
    data = data_processor.load_and_preprocess()
    scaled_data = data_processor.scale_data()
    logger.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    # Create sequences
    seq_length = 12  # Reduced from 24 to 12 hours
    logger.info(f"Creating sequences with length {seq_length}...")
    start_time = time.time()
    X, y = data_processor.create_sequences(seq_length)
    logger.info(f"Sequence creation completed in {time.time() - start_time:.2f} seconds")
    
    # Split data
    data_splits = data_processor.split_data(X, y)
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['val']
    X_test, y_test = data_splits['test']
    
    # Build and train model
    logger.info("Building and training model...")
    model = PowerConsumptionModel(input_shape=(seq_length, X_train.shape[2]))
    model.build_model()
    
    start_time = time.time()
    history = model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    
    # Save the trained model
    logger.info("Saving the trained model...")
    model.model.save('best_model_t.keras')
    logger.info("Model saved successfully as 'best_model_t'")
    
    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate and display metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred)
    logger.info(f"Model metrics: {metrics}")
    
    # Create visualizations
    evaluator.plot_predictions(y_test, y_pred)
    evaluator.plot_residuals(y_test, y_pred)
    evaluator.plot_error_distribution(y_test, y_pred)
    evaluator.plot_learning_curve(history)
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(y_test, y_pred)
    logger.info("\nEvaluation Report:")
    logger.info(report)
    
if __name__ == "__main__":
    main() 