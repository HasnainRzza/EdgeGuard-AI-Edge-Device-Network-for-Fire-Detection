import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model_factory import build_model
from model_trainer import train_model

@patch('model_trainer.mlflow')
def test_model_trainer(mock_mlflow, tmp_path):
    # Setup mock data (10 samples, 128x128x3)
    X_train = np.random.rand(10, 128, 128, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(10,))
    X_val = np.random.rand(4, 128, 128, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(4,))
    
    # Build a tiny model
    model = build_model()
    
    # Override config epochs for the test so it runs fast
    with patch('model_trainer.config') as mock_config:
        mock_dict = {
            'training': {
                'epochs': 1, 'batch_size': 2, 'learning_rate': 0.001,
                'img_size': 128, 'alpha': 0.25, 'classes': ['fire', 'no_fire']
            },
            'mlflow': {
                'experiment_name': 'test_exp',
                'tracking_uri': 'test_uri'
            }
        }
        mock_config.__getitem__.side_effect = lambda key: mock_dict.get(key, {})

        # Train model
        trained_model, model_path = train_model(model, X_train, y_train, X_val, y_val)
        
        # Verify
        assert trained_model is not None
        assert os.path.exists(model_path)
        
        # Verify MLflow was called
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.set_experiment.called
        assert mock_mlflow.start_run.called
        assert mock_mlflow.log_param.called
        assert mock_mlflow.log_metric.called
