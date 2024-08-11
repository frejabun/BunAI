import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import time
import shutil

model_list = [
    ("microsoft/Phi-3-mini-128k-instruct", "A small language model for experimentation"),
    ("cognitivecomputations/dolphin-2.9.4-llama3.1-8b", "A larger language model with improved capabilities")
]

custom_model_dir = "../BunAI/Models"

st.title("Model Selector")

selected_model = st.selectbox("Select a model", [model[0] for model in model_list])
selected_model_index = model_list.index(selected_model)
model_description = model_list[selected_model_index][1]
st.write(f"Model Description: {model_description}")

model_path = None

if st.button("Download Model"):
    progress_bar = st.progress(0)
    try:
        temp_model_path = hf_hub_download(repo_id=selected_model, filename="pytorch_model.bin")
        for i in range(100):
            time.sleep(0.01)  # Replace with actual download progress tracking
            progress_bar.progress(i + 1)

        # Create the custom directory if it doesn't exist
        import os
        os.makedirs(custom_model_dir, exist_ok=True)

        # Move the downloaded model to the custom directory
        shutil.move(temp_model_path, os.path.join(custom_model_dir, selected_model))
        model_path = os.path.join(custom_model_dir, selected_model)
    except Exception as e:
        st.error(f"Download failed: {e}")
        model_path = None

if model_path:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # ... rest of your code to use the model
