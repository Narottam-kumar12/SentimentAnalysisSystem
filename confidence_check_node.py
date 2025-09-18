class ConfidenceCheckNode:
    def __init__(self, threshold=0.7):
        self.threshold = threshold

    def is_confident(self, confidence: float) -> bool:
        return confidence >= self.threshold
