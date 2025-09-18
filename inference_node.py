import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from config import MODEL_NAME_OR_PATH, DEVICE, MAX_LENGTH

class InferenceNode:
  def __init__(self, model_path=MODEL_NAME_OR_PATH):
    self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
    self.model.to(DEVICE)

    self.ambiguous_phrases = {
        "okay", "fine", "not bad", "so-so", "meh",
        "could be better", "nothing special", "average", "mediocre"
    }
  def is_ambiguous(self, text: str) -> bool:
      text_lower = text.lower()
      return any(phrase in text_lower for phrase in self.ambiguous_phrases)


  def classify(self, text: str):
    # Tokenize and forward
    inputs = self.tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = self.model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred = int(torch.argmax(probs, dim=-1).cpu().numpy()[0])
    confidence = float(probs[0][pred].cpu().numpy())

    # Lower confidence heuristically for ambiguous phrases
    if self.is_ambiguous(text):
        confidence = min(confidence, 0.6)

    # Map to labels
    sentiment_label = "Positive" if pred == 1 else "Negative"
    return {
        "label": sentiment_label,
        "score": confidence,
        "raw_class": pred
    }
