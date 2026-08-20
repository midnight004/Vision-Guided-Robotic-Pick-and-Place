"""
Logging utility for the Vision-Guided Robotic Pick-and-Place system.
Provides structured, colored logging for each module.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


# Color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET": "\033[0m",
}

MODULE_COLORS = {
    "simulation": "\033[94m",   # Light blue
    "camera": "\033[96m",       # Light cyan
    "detection": "\033[93m",    # Light yellow
    "tracking": "\033[95m",     # Light magenta
    "localization": "\033[92m", # Light green
    "robot_control": "\033[91m",# Light red
    "task_logic": "\033[97m",   # White
    "evaluation": "\033[90m",   # Gray
}


class ColorFormatter(logging.Formatter):
    """Custom formatter with color support."""
    
    def format(self, record):
        level_color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        
        # Format: [TIME] [LEVEL] [MODULE] message
        record.levelname = f"{level_color}{record.levelname:8s}{reset}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: str = "results/logs"
) -> logging.Logger:
    """
    Set up a logger for a module.
    
    Args:
        name: Module name (e.g., 'detection', 'localization')
        level: Logging level
        log_to_file: Whether to also log to file
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"vision_robot.{name}")
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = ColorFormatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (no color)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name}_{timestamp}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
