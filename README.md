# Sentiment Analysis System

A self-healing sentiment classification service built on a fine-tuned DistilBERT model, with confidence-based fallback routing and a continuous feedback loop for improving prediction quality over time.

## Overview

Most sentiment classifiers fail silently on ambiguous or low-confidence inputs, returning a label without signaling uncertainty. This system addresses that gap by combining transformer-based classification with an explicit confidence-checking layer: predictions below a configurable confidence threshold are routed to a fallback path for clarification, and the resulting corrections are logged to support iterative model improvement.

The system exposes both a REST API for programmatic integration and a Streamlit interface for interactive testing.

## Architecture

```
                 ┌──────────────────┐
   Input Text →  │  Inference Node   │  (DistilBERT)
                 └────────┬─────────┘
                          │
                 ┌────────▼─────────┐
                 │ Confidence Check  │
                 └────────┬─────────┘
                          │
           ┌──────────────┴───────────────┐
           │                               │
   High confidence                  Low confidence
           │                               │
   ┌───────▼───────┐              ┌────────▼─────────┐
   │ Return result  │              │  Fallback Node    │
   └────────────────┘              │ (clarification /  │
                                    │  human-in-loop)   │
                                    └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  Feedback Logger  │
                                    └────────────────────┘
```

Each stage is implemented as an isolated module (`inference_node.py`, `confidence_check_node.py`, `fallback_node.py`), making the pipeline straightforward to test, extend, or swap components in independently — for example, replacing the underlying model without touching the confidence-routing logic.

## Key Design Decisions

**Confidence thresholding over raw classification.** Returning a label alone hides model uncertainty. Probability-based thresholding surfaces ambiguous cases explicitly rather than letting them fail silently downstream.

**Fallback as a first-class path, not an afterthought.** Low-confidence predictions are routed through a dedicated node rather than handled as an exception case, keeping the uncertainty-handling logic testable and observable.

**Structured logging for the feedback loop.** All predictions, confidence scores, and user corrections are logged in structured JSON rather than plain text, so the data is directly usable for future fine-tuning or threshold calibration without a separate parsing step.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![REST API](https://img.shields.io/badge/REST%20API-FF6C37?style=for-the-badge&logo=fastapi&logoColor=white)
![JSON](https://img.shields.io/badge/JSON%20Logging-000000?style=for-the-badge&logo=json&logoColor=white)

| Layer | Technology |
|---|---|
| Model | DistilBERT (fine-tuned, binary sentiment classification) |
| ML Framework | PyTorch, Hugging Face Transformers |
| Backend | REST API (`api.py`) |
| Frontend | Streamlit (`app_streamlit.py`) |
| Logging | Custom structured JSON logger |

## Project Structure

```
SentimentAnalysisSystem/
├── app_streamlit.py           # Streamlit UI
├── api.py                     # REST API for sentiment inference
├── inference_node.py          # Core DistilBERT prediction logic
├── confidence_check_node.py   # Confidence thresholding
├── fallback_node.py           # Low-confidence routing and clarification
├── custom_logger.py           # Structured logging utility
├── config.py                  # Configuration settings
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd SentimentAnalysisSystem

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

**Streamlit UI:**
```bash
streamlit run app_streamlit.py
```

**REST API:**
```bash
python api.py
```

### Example

**Request**
```
"I really enjoyed the movie, it was fantastic!"
```

**Response**
```json
{
  "sentiment": "Positive",
  "confidence": 0.99
}
```

## Future Improvements

- Extend beyond binary classification to a neutral/multi-class label set
- Add automated periodic re-training using logged feedback data
- Containerize the API for consistent deployment (Docker)
- Add latency and confidence-distribution monitoring dashboards

## License

MIT
