import logging

def setup_logger():
    """Configures and returns a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Output logs to console
        ]
    )
    return logging.getLogger(__name__)