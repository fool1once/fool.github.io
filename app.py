import streamlit as st
import requests

# -----------------------
# SETUP API
# -----------------------

API_URL = "https://api-inference.huggingface.co/models/Vamsi/T5_Paraphrase"

# IMPORTANT:
# Replace "YOUR_HF_API_TOKEN" with your real HuggingFace API token
HEADERS = {
    "Authorization": "Bearer YOUR_HF_API_TOKEN"
}

# -----------------------
# PARAPHRASING FUNCTION
# -----------------------
def paraphrase_text(text):
    try:
        payload = {
            "inputs": f"paraphrase: {text}",
            "parameters": {"temperature": 0.7}
        }

        response = requests.post(API_URL, headers=HEADERS, json=payload)

        if response.status_code != 200:
            return f"API Error: {response.text}"

        data = response.json()

        # HuggingFace returns a list of dictionaries
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]

        return "Unexpected response format."

    except Exception as e:
        return f"Error: {e}"

# -----------------------
# STREAMLIT UI
# -----------------------
def main():
    st.title("AI Human Paraphraser")
    st.write("Transform your text into natural, human-like writing — powered by AI.")

    text = st.text_area("Enter text to paraphrase:", height=200)

    if st.button("Paraphrase Text"):
        if not text.strip():
            st.error("Please enter some text first!")
        else:
            with st.spinner("Paraphrasing... please wait"):
                result = paraphrase_text(text)

            st.subheader("Paraphrased Output:")
            st.write(result)


if __name__ == "__main__":
    main()
