import streamlit as st
import requests
from config import CONFIDENCE_THRESHOLD

API_URL = "http://127.0.0.1:5001"
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("🧠 Sentiment Analysis — DistilBERT + Self-heal")

st.markdown("Enter text below and click **Analyze**. If the model is uncertain, you can correct it.")

text = st.text_area("Enter review", height=150)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please enter a review text.")
    else:
        try:
            res = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10).json()

        except Exception as e:
            st.error(f"Error calling API: {e}")
            st.stop()

        label = res.get("label")
        confidence = res.get("confidence")
        low_conf = res.get("low_confidence", False)

        st.subheader("Prediction")
        st.write(f"**Label:** {label}")
        st.write(f"**Confidence:** {confidence:.2f}")

        if low_conf:
            st.warning("⚠️ Low confidence — please verify the sentiment below")
            verified = st.radio("Was the prediction correct?", ("Yes", "No"))

            if verified == "No":
                corrected = st.selectbox("Correct sentiment", ("Positive", "Negative"))
                if st.button("Submit Correction"):
                    payload = {
                        "text": text,
                        "model_label": label,
                        "corrected_label": corrected
                    }
                    try:
                        fb = requests.post(f"{API_URL}/feedback", json=payload, timeout=10).json()
                        st.success(f"Feedback saved. Final label: {fb.get('final_label')}")
                    except Exception as e:
                        st.error(f"Failed to send feedback: {e}")

                else:
                    st.info("Thanks for verifying — no action needed.")









