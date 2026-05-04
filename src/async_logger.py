import asyncio
import json
import os
import time
from datetime import datetime

class AsyncBatchLogger:
    def __init__(self, batch_size=20, time_limit_sec=5.0, log_file="logs/api_requests.jsonl"):
        self.batch_size = batch_size
        self.time_limit_sec = time_limit_sec
        self.log_file = log_file
        self.queue = asyncio.Queue()
        self.flush_event = asyncio.Event()
        self.flush_task = None
        
    def start(self):
        """Starts the background worker for flushing logs."""
        # Ensure log directory exists relative to project root
        os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
        self.flush_task = asyncio.create_task(self._worker())

    async def stop(self):
        """Stops the worker and flushes any remaining logs."""
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        # Final flush on shutdown
        await self._flush_current_queue()

    async def log(self, record: dict):
        """Adds a log record to the queue asynchronously."""
        # Add timestamp if not present
        if "timestamp" not in record:
            from datetime import timezone
            record["timestamp"] = datetime.now(timezone.utc).isoformat()

            
        self.queue.put_nowait(record)
        # If batch size reached, signal the worker to flush immediately
        if self.queue.qsize() >= self.batch_size:
            self.flush_event.set()

    async def _worker(self):
        while True:
            try:
                # Wait until event is set (batch_size reached) OR time_limit_sec expires
                try:
                    await asyncio.wait_for(self.flush_event.wait(), timeout=self.time_limit_sec)
                except asyncio.TimeoutError:
                    pass # time limit expired
                
                self.flush_event.clear()
                
                # Check if there's anything to flush
                if not self.queue.empty():
                    await self._flush_current_queue()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in async logger worker: {e}")

    async def _flush_current_queue(self):
        batch = []
        while not self.queue.empty() and len(batch) < self.batch_size:
            try:
                record = self.queue.get_nowait()
                batch.append(record)
            except asyncio.QueueEmpty:
                break
        
        if not batch:
            return
            
        success = await self._write_to_storage_async(batch)
        if success:
            for _ in range(len(batch)):
                self.queue.task_done()
        else:
            # If flush fails, retry without losing logs by putting them back
            for record in batch:
                self.queue.put_nowait(record)
            # Add a small delay before retrying to prevent tight loop failures
            await asyncio.sleep(1.0)

    def _write_sync(self, batch):
        """Synchronous write function to be run in a separate thread."""
        try:
            with open(self.log_file, mode='a', encoding='utf-8') as f:
                content = "".join([json.dumps(record) + "\n" for record in batch])
                f.write(content)
            return True
        except Exception as e:
            print(f"Failed to write logs to {self.log_file}: {e}")
            return False

    async def _write_to_storage_async(self, batch):
        """Writes logs to disk asynchronously using a thread pool."""
        # asyncio.to_thread runs the synchronous function in a separate thread,
        # preventing it from blocking the main asyncio event loop.
        return await asyncio.to_thread(self._write_sync, batch)
