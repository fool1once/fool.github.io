import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Human Paraphraser",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    .stTextArea {
        border-radius: 10px;
    }
    .paraphrase-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .paraphrase-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource(show_spinner="Loading AI model... This may take a moment.")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase")
    return tokenizer, model

# Paraphrasing function
def paraphrase_text(text, tokenizer, model, max_length=512, num_beams=5, temperature=0.7):
    try:
        # Prepare input
        encoded = tokenizer.encode_plus(
            f"paraphrase: {text}",
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate paraphrase
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
        
        # Decode and return
        paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrase
    except Exception as e:
        return f"Error: {str(e)}"

# Human-likeness scoring
def human_score(text):
    score = 0
    
    # Vary sentence length
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) > 1:
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(set(sentence_lengths)) > 1:
            score += 10
    
    # Include transition words
    transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently', 'meanwhile', 'additionally']
    if any(t in text.lower() for t in transitions):
        score += 10
    
    # Avoid repetitive patterns
    words = text.lower().split()
    unique_words = set(words)
    if len(words) > 0 and len(unique_words) / len(words) > 0.7:
        score += 10
    
    return score

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">✍️ AI Human Paraphraser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform any text into natural, human-like writing</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Style options
        style = st.selectbox(
            "Paraphrasing Style",
            ["Balanced", "Formal", "Casual", "Creative"],
            index=0
        )
        
        # Complexity level
        complexity = st.selectbox(
            "Complexity Level",
            ["Simple", "Moderate", "Advanced"],
            index=1
        )
        
        # Advanced settings
        st.subheader("Advanced Settings")
        temperature = st.slider("Creativity", 0.1, 1.0, 0.7, 0.1)
        num_beams = st.slider("Quality", 1, 10, 5, 1)
        
        # Model info
        st.subheader("📊 Model Info")
        st.info("Using T5 Paraphrase model optimized for human-like text generation")
    
    # Main content area
    # Input section
    st.subheader("📝 Original Text")
    input_text = st.text_area(
        "Enter your text here...",
        height=200,
        placeholder="Paste or type the text you want to paraphrase...",
        help="The AI will rewrite this text to sound more natural and human-like."
    )
    
    # Character count
    if input_text:
        char_count = len(input_text)
        st.caption(f"Characters: {char_count}")
    
    # Paraphrase button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Paraphrase Text", type="primary", use_container_width=True):
            if not input_text:
                st.error("Please enter some text to paraphrase!")
            else:
                # Show loading
                with st.spinner("🤖 AI is working its magic..."):
                    # Load model
                    tokenizer, model = load_model()
                    
                    # Paraphrase
                    paraphrased = paraphrase_text(
                        input_text, tokenizer, model, 
                        temperature=temperature, 
                        num_beams=num_beams
                    )
                    
                    # Calculate human score
                    score = human_score(paraphrased)
                    
                    # Display results
                    st.subheader("✨ Paraphrased Text")
                    st.success(paraphrased)
                    
                    # Human-likeness score
                    st.metric("Human-likeness Score", f"{score}/30")
                    
                    # Copy button
                    if st.button("📋 Copy to Clipboard"):
                        st.toast("Text copied to clipboard!", icon="✅")
    
    # Example section
    with st.expander("📖 See Examples"):
        examples = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the way we work and live.",
            "Climate change is one of the most pressing challenges of our time."
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"Example {i}", key=f"example_{i}"):
                st.session_state.input_text = example
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by advanced AI technology | Built with Streamlit"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
