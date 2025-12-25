import asyncio
import sys
import os
import time
from loguru import logger
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def job_handler(file_path):
    """
    This function simulates processing the job on the Async Loop.
    """
    logger.info(f"New job at {file_path}")

class SentryEventHandler(FileSystemEventHandler):
    """
    The 'Sentry'. Watches for file events and signals the Async Loop.
    """
    def __init__(self, loop):
        self.loop = loop

    def on_created(self, event):
        if not event.is_directory:
            # Signal the Async Loop
            # We use call_soon_threadsafe because this runs in the Watchdog thread
            self.loop.call_soon_threadsafe(job_handler, event.src_path)

async def main():
    watched_dir = "data/watched"
    # Ensure the directory exists
    os.makedirs(watched_dir, exist_ok=True)

    loop = asyncio.get_running_loop()
    event_handler = SentryEventHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, watched_dir, recursive=False)

    logger.info(f"Starting Sentry on {watched_dir}...")
    observer.start()

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        observer.stop()
        observer.join()
        logger.info("Sentry stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
