import streamlit as st
import requests

# Correct HuggingFace Router endpoint
API_URL = "https://router.huggingface.co/pipeline/text2text-generation/Vamsi/T5_Paraphrase"

# Replace this with your real HF API token
HEADERS = {
    "Authorization": f"Bearer {st.secrets['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']}"
}

def paraphrase_text(text):
    try:
        payload = {
            "inputs": f"paraphrase: {text}",
            "parameters": {
                "temperature": 0.7
            }
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            return f"API Error: {response.text}"

        data = response.json()

        # Expected format:
        # [{"generated_text": "..."}]
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        return "Unexpected response format."

    except Exception as e:
        return f"Error: {e}"

def main():
    st.title("AI Human Paraphraser")
    st.write("Transform your text into natural, human-like writin
