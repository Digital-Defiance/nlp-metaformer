
import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger