# Watchdog Sentry

A Python project that monitors a directory for new PDF files and processes them asynchronously.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Directory Structure:**
    Ensure the following directories exist (created automatically by the script):
    *   `data/watched`: The folder being monitored.
    *   `data/samples`: Contains sample PDF files.

## Usage

### Starting the Sentry

Run the Sentry module from the root directory:

```bash
python3 -m src.sentry
```

You will see a log message indicating that the Sentry has started.

### Testing

To trigger a job, copy a file into the watched directory while the Sentry is running:

```bash
cp data/samples/2312.01797v3.pdf data/watched/test_job.pdf
```

Check the Sentry's output. You should see a log message confirming the new job:

```
... | INFO | src.simulator:job_handler:7 - New job at data/watched/test_job.pdf
```

## Architecture

For a detailed explanation of the system architecture, please refer to [ARCHITECTURE.md](ARCHITECTURE.md).
