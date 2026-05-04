import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data_ingestion import load_dataset

def test_load_dataset(tmp_path):
    # Setup mock dataset structure
    fire_dir = tmp_path / "fire_images"
    non_fire_dir = tmp_path / "non_fire_images"
    fire_dir.mkdir()
    non_fire_dir.mkdir()
    
    # Create mock images (empty files are enough for ingestion logic, it only checks extensions)
    (fire_dir / "fire1.jpg").touch()
    (fire_dir / "fire2.png").touch()
    (fire_dir / ".hidden").touch() # Should be ignored
    (fire_dir / "not_image.txt").touch() # Should be ignored
    
    (non_fire_dir / "nofire1.jpg").touch()
    
    X, y = load_dataset(str(tmp_path))
    
    assert len(X) == 3
    assert len(y) == 3
    assert list(y) == [0, 0, 1]
    
    assert any("fire1.jpg" in str(p) for p in X)
    assert any("not_image.txt" in str(p) for p in X) is False
