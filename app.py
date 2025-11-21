import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model with caching
@st.cache_resource(show_spinner="Loading AI model... This may take a moment.")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase")
    return tokenizer, model

# Paraphrasing function
def paraphrase_text(text, tokenizer, model, max_length=512, num_beams=5, temperature=0.7):
    try:
        encoded = tokenizer.encode_plus(
            f"paraphrase: {text}",
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        outputs = model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            early_stopping=True
        )

        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrase
    except Exception as e:
        return f"Error: {str(e)}"

# Main app
def main():
    st.title("AI Human Paraphraser")
    st.write("Transform any text into natural, human-like writing")

    input_text = st.text_area("Enter your text here...", height=200)

    if st.button("Paraphrase Text"):
        if not input_text:
            st.error("Please enter some text to paraphrase!")
        else:
            tokenizer, model = load_model()
            paraphrased = paraphrase_text(input_text, tokenizer, model)
            st.success("Paraphrased Text:")
            st.write(paraphrased)

if __name__ == "__main__":
    main()
