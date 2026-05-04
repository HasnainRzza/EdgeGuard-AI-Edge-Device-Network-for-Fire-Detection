import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from pipeline import run_pipeline

@patch('pipeline.convert_to_tflite')
@patch('pipeline.log_to_prometheus')
@patch('pipeline.analyze_performance')
@patch('pipeline.evaluate_model')
@patch('pipeline.train_model')
@patch('pipeline.build_model')
@patch('pipeline.get_augmentation')
@patch('pipeline.transform_data')
@patch('pipeline.train_test_split')
@patch('pipeline.load_dataset')
@patch('pipeline.get_data_ingestion_config')
def test_run_pipeline(
    mock_get_config, mock_load, mock_split, mock_transform, mock_aug, 
    mock_build, mock_train, mock_eval, mock_analyze, mock_log, mock_convert
):
    # Setup mocks
    mock_get_config.return_value = "dummy_path"
    mock_load.return_value = (["img1", "img2"], [0, 1])
    
    # train_test_split is called twice. 
    # Call 1: X_train, X_temp, y_train, y_temp
    # Call 2: X_val, X_test, y_val, y_test
    mock_split.side_effect = [
        (["img1"], ["img2"], [0], [1]),
        (["img2"], ["img3"], [1], [0])
    ]
    
    # transform_data returns (X, y)
    mock_transform.return_value = (MagicMock(), MagicMock())
    
    mock_aug.return_value = MagicMock()
    
    mock_model = MagicMock()
    mock_model.count_params.return_value = 100
    mock_build.return_value = mock_model
    
    mock_train.return_value = (mock_model, "dummy_model_path")
    
    mock_eval.return_value = {"accuracy": 0.9}
    
    mock_convert.return_value = "dummy_tflite_path"
    
    # Execute pipeline
    run_pipeline()
    
    # Assert orchestrator logic
    assert mock_get_config.called
    mock_load.assert_called_with("dummy_path")
    assert mock_split.call_count == 2
    assert mock_transform.call_count == 3
    assert mock_build.called
    assert mock_train.called
    assert mock_eval.called
    assert mock_analyze.called
    mock_log.assert_called_with({"accuracy": 0.9, "total_params": 100})
    assert mock_convert.called
