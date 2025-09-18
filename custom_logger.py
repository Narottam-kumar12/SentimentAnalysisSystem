import logging
import json
from config import LOG_FILE
from datetime import datetime
import os

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logger = logging.getLogger("sentiment_logger")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def log_event(event: dict):
    event_copy = event.copy()
    event_copy["timestamp"] = datetime.utcnow().isoformat()
    logger.info(json.dumps(event_copy, ensure_ascii=False))

