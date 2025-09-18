from flask import Flask, request, jsonify
from inference_node import InferenceNode
from confidence_check_node import ConfidenceCheckNode
from fallback_node import FallbackNode
from custom_logger import log_event
from config import CONFIDENCE_THRESHOLD
app = Flask(__name__)
# Instantiate components (singletons)
inference = InferenceNode()
confidence_checker = ConfidenceCheckNode(CONFIDENCE_THRESHOLD)
fallback = FallbackNode()
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' in JSON body"}), 400

    text = data["text"]
    res = inference.classify(text)
    label = res["label"]
    score = res["score"]

    # Log initial prediction
    log_event({
        "event": "prediction",
        "text": text,
        "model_label": label,
        "confidence": score
    })

    # If low confidence, return fallback suggestion
    if not confidence_checker.is_confident(score):
        return jsonify({
            "label": label,
            "confidence": score,
            "low_confidence": True,
            "fallback": fallback.prepare_fallback(text, label, score)
        })

    return jsonify({"label": label, "confidence": score, "low_confidence": False})

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    # expected: { text, model_label, corrected_label }
    text = data.get("text")
    model_label = data.get("model_label")
    corrected_label = data.get("corrected_label")

    if not text or not model_label or not corrected_label:
        return jsonify({"error": "Missing fields (text, model_label, corrected_label)"}), 400

    # Apply correction logic if needed (here we simply accept corrected_label)
    final_label = fallback.apply_correction(corrected_label)

    # Log correction
    log_event({
        "event": "correction",
        "text": text,
        "model_label": model_label,
        "corrected_label": final_label
    })

    return jsonify({"status": "ok", "final_label": final_label})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)


