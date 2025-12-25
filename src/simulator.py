from loguru import logger

def job_handler(file_path):
    """
    This function simulates processing the job on the Async Loop.
    """
    logger.info(f"New job at {file_path}")
