import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_transformation import transform_data

def test_transform_data(tmp_path):
    # Create two actual valid images
    img1_path = str(tmp_path / "img1.jpg")
    img2_path = str(tmp_path / "img2.jpg")
    img_bad_path = str(tmp_path / "img_bad.jpg")
    
    # Create valid 10x10 dummy images
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imwrite(img1_path, dummy_img)
    cv2.imwrite(img2_path, dummy_img)
    
    # Create a corrupted image file
    with open(img_bad_path, "wb") as f:
        f.write(b"not an image")
        
    X_paths = np.array([img1_path, img_bad_path, img2_path])
    y = np.array([0, 1, 0])
    
    X_trans, y_trans = transform_data(X_paths, y)
    
    # The bad image should be dropped, preserving the rest
    assert len(X_trans) == 2
    assert len(y_trans) == 2
    assert list(y_trans) == [0, 0]
    
    # Verify shape and normalization
    # shape should be (2, 128, 128, 3) because data_transformation resizes to config['training']['img_size']
    assert X_trans.shape[1:] == (128, 128, 3)
    assert X_trans.dtype == np.float32
    assert np.max(X_trans) <= 1.0
    assert np.min(X_trans) >= 0.0
