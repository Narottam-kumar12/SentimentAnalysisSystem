MODEL_NAME_OR_PATH = "distilbert-base-uncased-finetuned-sst-2-english"
CONFIDENCE_THRESHOLD = 0.7
LOG_FILE = "logs/classification_log.txt"
MAX_LENGTH = 512
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"