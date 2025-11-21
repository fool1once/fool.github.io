from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)
CORS(app)

class HumanParaphraser:
    def __init__(self):
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase")
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase")
    
    def paraphrase(self, text, max_length=512):
        try:
            encoded = self.tokenizer.encode_plus(
                f"paraphrase: {text}",
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            
            outputs = self.model.generate(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                max_length=max_length,
                num_beams=5,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return paraphrase
        except Exception as e:
            return f"Error: {str(e)}"

paraphraser = HumanParaphraser()

@app.route('/paraphrase', methods=['POST'])
def paraphrase_endpoint():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    paraphrased = paraphraser.paraphrase(text)
    
    return jsonify({
        'original': text,
        'paraphrased': paraphrased
    })

@app.route('/')
def home():
    return "AI Paraphraser API is running!"

if __name__ == '__main__':
    app.run(debug=True)
