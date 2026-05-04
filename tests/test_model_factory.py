import os
import sys
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_factory import build_model

def test_build_model():
    # Build model (uses config values internally)
    model = build_model()
    
    # Assert type
    assert isinstance(model, tf.keras.Model)
    
    # Assert input shape
    assert model.input_shape == (None, 128, 128, 3)
    
    # Assert output shape (2 classes)
    assert model.output_shape == (None, 2)
    
    # Assert it compiles and has the correct loss
    assert model.loss == 'sparse_categorical_crossentropy'
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    
    # Assert parameter count is reduced due to alpha 0.5 (should be roughly ~40k)
    assert model.count_params() < 100000
