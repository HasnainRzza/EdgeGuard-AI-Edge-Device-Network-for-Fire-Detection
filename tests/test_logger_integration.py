import asyncio
import os
import sys
import json
import pytest

# Ensure src/ is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from async_logger import AsyncBatchLogger
from utils import get_project_root

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.mark.anyio
async def test_async_logger():
    # Setup test file
    log_file = os.path.join(get_project_root(), "logs", "test_api_requests.jsonl")
    if os.path.exists(log_file):
        os.remove(log_file)
        
    logger = AsyncBatchLogger(batch_size=5, time_limit_sec=1.0, log_file=log_file)
    logger.start()
    
    # Send 4 logs (less than batch size)
    for i in range(4):
        await logger.log({"request_id": i, "endpoint": "/test"})
        
    # Wait slightly to ensure they aren't flushed before time limit
    await asyncio.sleep(0.1)
    
    # Should not exist yet, or be empty
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 0, "Logs flushed too early"
            
    # Send 1 more log to hit batch size
    await logger.log({"request_id": 4, "endpoint": "/test"})
    
    # Wait for the event to trigger flush
    await asyncio.sleep(0.5)
    
    assert os.path.exists(log_file), "Log file was not created"
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 5, f"Expected 5 logs, got {len(lines)}"
        
    # Test time limit flush
    await logger.log({"request_id": 5, "endpoint": "/test_time_limit"})
    
    # Wait for time limit to expire (1.0 sec)
    await asyncio.sleep(1.5)
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 6, f"Expected 6 logs after time limit flush, got {len(lines)}"
        
    await logger.stop()
    
    # Cleanup
    if os.path.exists(log_file):
        os.remove(log_file)
