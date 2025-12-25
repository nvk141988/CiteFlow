# System Architecture

This project implements a file monitoring system that bridges filesystem events to an asynchronous event loop.

## Components

### 1. Sentry (`src/sentry.py`)
The **Sentry** is the entry point of the application. It has two main roles:
*   **Async Loop Host:** It starts and runs the `asyncio` main event loop.
*   **Watchdog Observer:** It configures and starts a `watchdog.Observer` to monitor the `data/watched/` directory.

The Sentry uses a custom `FileSystemEventHandler` (`SentryEventHandler`) to react to file creation events. Crucially, this handler is designed to do **zero processing** on the Watchdog thread. Instead, it immediately offloads the work to the main async loop.

### 2. Simulator (`src/simulator.py`)
The **Simulator** contains the business logic for processing jobs. Currently, it implements `job_handler(file_path)`, which logs the detection of a new file. This represents the "consumer" side of the system.

## Data Flow

1.  **File Creation:** A file is dropped into `data/watched/`.
2.  **Detection:** The OS notifies `watchdog`, which calls `SentryEventHandler.on_created` on a background thread.
3.  **Signal:** The handler calls `loop.call_soon_threadsafe(job_handler, file_path)`. This schedules the execution of `job_handler` on the main thread's asyncio loop.
4.  **Processing:** The `asyncio` loop picks up the scheduled callback and executes `job_handler`, ensuring thread safety and allowing for non-blocking operations in the future.

## Directory Structure

*   `src/`: Source code.
*   `data/watched/`: Directory monitored for new jobs.
*   `data/samples/`: Storage for sample data.
