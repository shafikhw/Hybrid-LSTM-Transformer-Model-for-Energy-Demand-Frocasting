import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    LayerNormalization, MultiHeadAttention, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PowerConsumptionModel:
    def __init__(self, input_shape: tuple, lstm_units: int = 64, dropout_rate: float = 0.2):
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        
        # Configure GPU
        self._configure_gpu()
        
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow."""
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
        
    def build_model(self) -> Model:
        """Build a simplified LSTM model."""
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First LSTM layer
        x = LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Second LSTM layer
        x = LSTM(self.lstm_units)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info("Model built successfully")
        return model
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 64,
              patience: int = 5) -> tf.keras.callbacks.History:
        """Train the model with early stopping and model checkpointing."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'lstm_best.keras',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Set device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Train on GPU
        with tf.device('/GPU:0'):
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        logger.info("Model training completed")
        return history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
            
        # Set device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Predict on GPU and ensure float32 output
        with tf.device('/GPU:0'):
            predictions = self.model.predict(X)
            # Convert to float32 to ensure consistency
            return predictions.astype(np.float32)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train first.")
            
        # Set device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Evaluate on GPU
        with tf.device('/GPU:0'):
            metrics = self.model.evaluate(X, y, return_dict=True)
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics 