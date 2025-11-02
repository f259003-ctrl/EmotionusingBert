import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import gdown
import os
import shutil

# -----------------------------
# CONFIG
# -----------------------------
DRIVE_FOLDER_ID = "1AK8V47qGumkdIWO8EWt-ZkPqMvMUACQp"  # from your shared folder link
MODEL_DIR = "bert_emotion_model"

# -----------------------------
# DOWNLOAD MODEL FROM DRIVE
# -----------------------------
@st.cache_resource
def download_model_from_drive():
    """
    Downloads model folder from Google Drive and extracts it locally.
    Assumes it was zipped before uploading.
    """
    if os.path.exists(MODEL_DIR):
        return MODEL_DIR

    zip_path = "model.zip"
    drive_url = f"https://drive.google.com/uc?id={DRIVE_FOLDER_ID}"
    st.info("📦 Downloading model from Google Drive...")
    gdown.download_folder(url=drive_url, output=MODEL_DIR, quiet=False, use_cookies=False)
    st.success("✅ Model downloaded successfully.")
    return MODEL_DIR


# -----------------------------
# LOAD MODEL AND TOKENIZER
# -----------------------------
@st.cache_resource
def load_model():
    model_path = download_model_from_drive()
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="BERT Emotion Detector", page_icon="🤖", layout="centered")
st.title("🤖 Emotion Detection using Fine-Tuned BERT")
st.markdown("Type a sentence below to detect its emotional tone:")

# Text input
user_input = st.text_area("Enter a sentence:", placeholder="e.g. I'm so happy today!")

if st.button("Analyze Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a sentence.")
    else:
        with st.spinner("Analyzing..."):
            # Tokenize input
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            )

            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred_label].item()

            id2label = model.config.id2label
            emotion = id2label[pred_label]

        # Display prediction
        st.success(f"**Predicted Emotion:** {emotion}")
        st.write(f"**Confidence:** {confidence:.2%}")

        # Show all emotion probabilities
        st.subheader("Emotion Probabilities")
        prob_dict = {id2label[i]: float(probs[0][i]) for i in range(len(probs[0]))}
        st.bar_chart(prob_dict)

st.markdown("---")
st.caption("🧩 Fine-tuned BERT model loaded from Google Drive and deployed using Streamlit.")
