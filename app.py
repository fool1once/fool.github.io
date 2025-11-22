import streamlit as st
import openai
import google.generativeai as genai
import re
import io
from datetime import datetime

# --- Configuration ---
# Load API keys from Streamlit secrets
try:
    openai.api_key = st.secrets["sk-proj-tikeIPr3HZmrLULa3K1GeLdkvZdrhxg_sKPbXZqWNroYYRH_zQRpO6wWF7Y-5KS6SYqazzPKjAT3BlbkFJJ743FHVM4ZxJRg1z1CKzIrAXyh7U9y2TY2fkhmom99wqdjdr-zVeXLHq5OB9YA8pVLMhi"]
    genai.configure(api_key=st.secrets["AIzaSyB5DEBrEq8RvEhtfSldxYn6OTJIaX7t6yg"])
    gemini_model = genai.GenerativeModel('gemini-pro')
except KeyError as e:
    st.error(f"⚠️ API Key Missing: {e}. Please add it to your Streamlit Cloud Secrets.")
    st.stop()

# --- AI Model Functions ---

def paraphrase_with_openai(text, creativity=0.7):
    """Paraphrases text using OpenAI's GPT-4 model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert writer. Rewrite the user's text to be completely natural and human-like, preserving the original meaning but changing the structure and vocabulary. Do not add any explanations, just return the rewritten text."},
                {"role": "user", "content": f"Rewrite this text: {text}"}
            ],
            temperature=creativity,
            max_tokens=2000 # Increased max length
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {e}"

def paraphrase_with_gemini(text):
    """Paraphrases text using Google's Gemini Pro model."""
    try:
        prompt = f"Paraphrase the following text to sound more natural and human-like. Maintain the original meaning. Do not add any explanations, just provide the rewritten text.\n\nOriginal Text: {text}\n\nParaphrased Text:"
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

# --- Enhanced Ensemble Logic ---

def human_score(text):
    """A more advanced scoring function to rate the 'human-like' quality of text."""
    score = 0
    words = text.lower().split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]

    if not words or not sentences:
        return 0

    # 1. Vocabulary Diversity (avoid repetition)
    unique_words = set(words)
    vocab_ratio = len(unique_words) / len(words)
    score += round(vocab_ratio * 20)

    # 2. Sentence Length Variation
    sentence_lengths = [len(s.split()) for s in sentences]
    avg_len = sum(sentence_lengths) / len(sentence_lengths)
    variance = sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
    if variance > 4: # Good variance
        score += 15

    # 3. Use of Transition Words (shows flow)
    transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'meanwhile', 'additionally', 'in fact', 'as a result', 'on the other hand']
    transition_count = sum(1 for t in transitions if t in text.lower())
    score += min(transition_count * 5, 15)

    # 4. Natural Language Patterns (contractions, common phrases)
    natural_patterns = ["it's", "don't", "you're", "won't", "can't", "i'm", "that's", "there's"]
    pattern_count = sum(1 for p in natural_patterns if p in text.lower())
    score += min(pattern_count * 3, 10)

    return min(score, 50) # Max score is 50

def ensemble_paraphrase(text):
    """Runs both models and returns the best-scoring result."""
    with st.spinner("Running GPT-4 and Gemini Pro... this may take up to 30 seconds."):
        results = []

        # Get results from each model
        openai_result = paraphrase_with_openai(text)
        gemini_result = paraphrase_with_gemini(text)

        results.append(("GPT-4", openai_result, human_score(openai_result)))
        results.append(("Gemini Pro", gemini_result, human_score(gemini_result)))

        # Find the result with the highest human_score
        best_result = max(results, key=lambda x: x[2])

        return best_result # Returns a tuple: (model_name, text, score)

# --- Main Streamlit App ---

def main():
    st.set_page_config(page_title="Advanced AI Paraphraser", page_icon="🧠", layout="centered")

    # Custom CSS for a cleaner look
    st.markdown("""
    <style>
        .main-header {font-size: 2.5rem; font-weight: 600; text-align: center; color: #1f77b4;}
        .sub-header {text-align: center; color: #666; margin-bottom: 2rem;}
        .stTextArea {border-radius: 10px;}
        .metric-card {text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 10px; margin: 1rem 0;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧠 Advanced AI Paraphraser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by top-tier AI models for superior, human-like results.</p>', unsafe_allow_html=True)

    # User input
    text_input = st.text_area("✍️ Enter your text below:", height=200, placeholder="Paste or type the text you want to paraphrase...")

    # Sidebar for advanced settings
    with st.sidebar:
        st.header("⚙️ Advanced Settings")
        creativity = st.slider("Creativity Level", 0.1, 1.0, 0.7, 0.1, help="Higher values make the text more creative and varied. Lower values stick closer to the original.")

        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("**GPT-4:** Excellent for creative and nuanced rewrites.\n\n**Gemini Pro:** Fast, balanced, and great value.")

    # Model selection
    st.markdown("---")
    st.subheader("🤖 Choose Your AI Engine")
    model_choice = st.radio(
        "Select the model to use:",
        ["GPT-4 (Creative & Nuanced)", "Gemini Pro (Fast & Balanced)", "🏆 Best of All (Ensemble)"],
        index=0
    )

    # Paraphrase button
    if st.button("🔄 Paraphrase Text", type="primary", use_container_width=True):
        if not text_input.strip():
            st.error("Please enter some text to paraphrase!")
        else:
            result_text = ""
            model_used = ""
            score = 0

            if "GPT-4" in model_choice:
                with st.spinner("GPT-4 is rewriting your text..."):
                    result_text = paraphrase_with_openai(text_input, creativity=creativity)
                model_used = "GPT-4"
                score = human_score(result_text
