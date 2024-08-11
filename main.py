import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_list = [
    ("microsoft/Phi-3-mini-128k-instruct", "A small language model for experimentation"),
    ("cognitivecomputations/dolphin-2.9.4-llama3.1-8b", "A larger language model with improved capabilities")
]

custom_model_dir = "../BunAI/Models"

st.title("Model Selector")

# Get list of model names for the dropdown
model_names = [model[0] for model in model_list]

selected_model = st.selectbox("Select a model", model_names)

# Find the selected model's description
model_description = None
for model_name, description in model_list:
    if selected_model == model_name:
        model_description = description
        break

if model_description:
    st.write(f"Model Description: {model_description}")

    model_save_path = os.path.join(custom_model_dir, selected_model.replace("/", "_"))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(selected_model, cache_dir=model_save_path)
        model.save_pretrained(model_save_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_save_path)

    # ... rest of your code to use the model
else:
    st.error(f"Invalid model selection: {selected_model}")


