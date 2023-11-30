import logging
import os
import sys  # Import sys module
from datetime import datetime
from logging.handlers import RotatingFileHandler
import traceback
import functools

def log_exception(exc_type, exc_value, exc_traceback):
    """
    Log exception with traceback.
    """
    logger.error("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))

def logging_decorator(func):
    """
    A decorator to log function entry, exit, and exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Entering: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"Exiting: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

def setup_logger(log_level=logging.DEBUG):
    """
    Set up a logger with a specified log level and rotating file handler.
    """
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{logs_dir}/redirect_matchmaker_log_{timestamp}.log"

    logger = logging.getLogger("RedirectMatchmakerLogger")
    logger.setLevel(log_level)

    file_handler = RotatingFileHandler(log_filename, maxBytes=10485760, backupCount=5)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.getLogger().addHandler(logging.StreamHandler())
    sys.excepthook = log_exception

    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logging setup complete.")
