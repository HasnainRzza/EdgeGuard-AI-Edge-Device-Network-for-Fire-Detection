import pytest
from fastapi.testclient import TestClient
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from api import app, pending_feedback, FEEDBACK_TTL
from utils import get_project_root
from unittest.mock import MagicMock, patch

client = TestClient(app)

def test_infer_low_confidence():
    # Mock model prediction to return low confidence
    mock_model = MagicMock()
    # 0.6 for class 0, 0.4 for class 1
    mock_model.predict.return_value = np.array([[0.6, 0.4]])
    
    from api import model_store
    model_store["model"] = mock_model
    
    # Create a dummy image
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    import cv2
    _, img_encoded = cv2.imencode('.jpg', img)
    
    response = client.post(
        "/infer", 
        files={"file": ("test.jpg", img_encoded.tobytes(), "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["needs_feedback"] is True
    assert "feedback_id" in data
    
    feedback_id = data["feedback_id"]
    assert feedback_id in pending_feedback
    
    return feedback_id

def test_feedback_endpoint_and_finetune():
    # First, get a feedback_id
    feedback_id = test_infer_low_confidence()
    
    # Ensure features directory is clean
    features_dir = os.path.join(get_project_root(), "features", "fire")
    if os.path.exists(features_dir):
        for f in os.listdir(features_dir):
            if feedback_id in f:
                os.remove(os.path.join(features_dir, f))
                
    # Submit feedback
    response = client.post("/feedback", json={
        "feedback_id": feedback_id,
        "label": "fire"
    })
    
    assert response.status_code == 200
    assert feedback_id not in pending_feedback
    
    # Verify files created (1 orig + 3 aug)
    files = [f for f in os.listdir(features_dir) if feedback_id in f and f.endswith(".npy")]
    assert len(files) == 4
    
    # Now trigger finetuning
    # Mock model fit
    from api import model_store
    mock_model = MagicMock()
    mock_model.fit.return_value.history = {"accuracy": [0.9], "loss": [0.1]}
    model_store["model"] = mock_model
    
    with patch('mlflow.set_tracking_uri'), \
         patch('mlflow.set_experiment'), \
         patch('mlflow.start_run'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.tensorflow.log_model'):
         
        response = client.post("/finetune")
        assert response.status_code == 200
        assert response.json()["message"] == "Finetuning started in the background."
    
    # Cleanup
    for f in files:
        os.remove(os.path.join(features_dir, f))
