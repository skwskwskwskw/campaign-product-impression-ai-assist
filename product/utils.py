"""
System worker configuration utilities with logging and memory management.
"""

import os
import multiprocessing
import logging
import gc
from typing import Dict, Optional


def setup_logging(level: str = "INFO", log_format: Optional[str] = None, enable_file_logging: bool = False, log_file: str = "product_matching.log"):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        enable_file_logging: Whether to also log to a file
        log_file: Path to the log file
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure the root logger
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler (optional)
    if enable_file_logging:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Apply handlers to root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        force=True  # Overwrite any existing configuration
    )


def get_system_workers_info():
    """Get system info for optimal worker configuration."""

    # CPU cores
    cpu_count = os.cpu_count() or 1

    # Multiprocessing cpu count (sometimes differs)
    mp_cpu_count = multiprocessing.cpu_count()

    # For I/O bound tasks (like URL processing), can use more than CPU count
    # ThreadPoolExecutor default is min(32, cpu_count + 4)
    default_thread_workers = min(32, cpu_count + 4)

    # For CPU bound tasks, use ProcessPoolExecutor with cpu_count
    recommended_process_workers = cpu_count

    # Conservative for mixed workloads
    conservative_workers = max(1, cpu_count - 1)

    info = {
        "cpu_count": cpu_count,
        "mp_cpu_count": mp_cpu_count,
        "default_thread_workers": default_thread_workers,
        "recommended_io_bound": default_thread_workers,  # URL fetching is I/O bound
        "recommended_cpu_bound": recommended_process_workers,
        "conservative": conservative_workers,
    }

    logging.info("=" * 50)
    logging.info("SYSTEM WORKER CONFIGURATION")
    logging.info("=" * 50)
    logging.info(f"CPU cores (os.cpu_count):        {cpu_count}")
    logging.info(f"CPU cores (multiprocessing):     {mp_cpu_count}")
    logging.info("-" * 50)
    logging.info(f"ThreadPoolExecutor default:      {default_thread_workers}")
    logging.info(f"Recommended for I/O bound:       {default_thread_workers}")
    logging.info(f"Recommended for CPU bound:       {recommended_process_workers}")
    logging.info(f"Conservative (leave 1 free):     {conservative_workers}")
    logging.info("=" * 50)

    return info


def get_optimal_workers(task_type: str = "io") -> int:
    """Get optimal worker count based on task type."""
    cpu_count = os.cpu_count() or 1

    if task_type == "io":
        return min(32, cpu_count + 4)
    elif task_type == "cpu":
        return cpu_count
    else:  # conservative
        return max(1, cpu_count - 1)


def memory_monitor():
    """Monitor memory usage and trigger garbage collection if needed."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logging.debug(f"Memory usage: RSS={memory_info.rss / 1024 / 1024:.2f} MB, "
                  f"VMS={memory_info.vms / 1024 / 1024:.2f} MB, "
                  f"Percent={memory_percent:.2f}%")

    # Trigger garbage collection if memory usage is high
    if memory_percent > 80:  # More than 80% memory usage
        collected = gc.collect()
        logging.info(f"Garbage collection triggered: {collected} objects collected")


def validate_dataframe(df, required_columns: list, df_name: str = "DataFrame"):
    """
    Validate that a DataFrame has the required columns.

    Args:
        df: The DataFrame to validate
        required_columns: List of required column names
        df_name: Name of the DataFrame for error messages

    Raises:
        ValueError: If required columns are missing
    """
    if df is None:
        raise ValueError(f"{df_name} is None")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{df_name} is missing required columns: {missing_cols}")

    logging.debug(f"{df_name} validation passed. Shape: {df.shape}, "
                  f"Columns: {list(df.columns)}")