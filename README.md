from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HumanParaphraser:
    def __init__(self):
        # Use models like T5, PEGASUS, or BART for human-like paraphrasing
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase")
        self.tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase")
    
    def paraphrase(self, text, max_length=512):
        # Multiple paraphrasing attempts for variety
        paraphrases = []
        for _ in range(3):
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
            paraphrases.append(paraphrase)
        
        return self.select_most_human(paraphrases)
    
    def select_most_human(self, paraphrases):
        # Score based on human-like characteristics
        scored = []
        for p in paraphrases:
            score = self.human_score(p)
            scored.append((score, p))
        
        return max(scored, key=lambda x: x[0])[1]
    
    def human_score(self, text):
        # Scoring algorithm for human-likeness
        score = 0
        
        # Vary sentence length
        sentences = text.split('.')
        sentence_lengths = [len(s.split()) for s in sentences if s]
        if len(set(sentence_lengths)) > 1:
            score += 10
        
        # Include transition words
        transitions = ['however', 'therefore', 'moreover', 'furthermore', 'consequently']
        if any(t in text.lower() for t in transitions):
            score += 10
        
        # Avoid repetitive patterns
        words = text.lower().split()
        unique_words = set(words)
        if len(unique_words) / len(words) > 0.7:
            score += 10
        
        return score
