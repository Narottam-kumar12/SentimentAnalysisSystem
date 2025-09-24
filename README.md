# 🧠 Sentiment Analysis System — DistilBERT + Self-Healing 
This project is a **Sentiment Analysis System** that classifies text into **Positive** or **Negative** sentiments using a **DistilBERT transformer model**.  
It also includes a **confidence-based fallback mechanism** and a **feedback loop** where users can correct low-confidence predictions, making the system more robust and adaptable over time.  

-----

## 🚀 Features  

- **Transformer-based model**: Uses Hugging Face’s **DistilBERT** for high-quality sentiment classification.  
- **Confidence checking**: Flags uncertain predictions based on probability and ambiguous phrases.  
- **Fallback system**: Routes low-confidence predictions to a clarification UI for human verification.  
- **Feedback loop**: User corrections are logged for future improvements.  
- **Interactive UI**: Built with **Streamlit** for easy testing.  
- **API Support**: Backend endpoints for integration with other applications.  
- **Custom logging**: Structured JSON logging for events, confidence, and feedback.  


## 🛠️ Tech Stack 

- **Language:** Python  
- **Libraries:** PyTorch, Transformers (Hugging Face), Streamlit, Requests, Logging  
- **Model:** DistilBERT (fine-tuned for binary sentiment classification)  
- **Backend:** REST API (`api.py`)  
- **Frontend:** Streamlit (`app_streamlit.py`)

## 📂 Project Structure 
SentimentAnalysisSystem/
│── app_streamlit.py # Streamlit user interface

│── api.py # API for sentiment inference

│── inference_node.py # Core sentiment prediction logic

│── fallback_node.py # Handles uncertain predictions

│── confidence_check_node.py# Confidence thresholding

│── config.py # Configuration settings

│── custom_logger.py # Logging utility

│── requirements.txt # Project dependencies

│── README.md # Project documentation

### Create virtual environment (recommended)
- python -m venv venv
- source venv/bin/activate   # For Linux/Mac  
- venv\Scripts\activate      # For Windows

### Install dependencies
- pip install -r requirements.txt

### 📊 Example
Input
- "I really enjoyed the movie, it was fantastic!"

output

- Sentiment: Positive ✅
- Confidence: 99%



