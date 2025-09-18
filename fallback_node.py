class FallbackNode:
    def prepare_fallback(self, text: str, model_label: str, confidence: float) -> dict:
        """
        Returns a dict that frontend can use to show clarification UI.
        """
        return {
            "text": text,
            "model_label": model_label,
            "confidence": confidence,
            "requires_human": True
        }

    def apply_correction(self, corrected_label: str) -> str:

       # In future: more complex mapping or label validation
       return corrected_label