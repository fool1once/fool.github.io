from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def paraphrase_with_local_model(text):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    prompt = f"[INST] Paraphrase this text to sound human-written: {text} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1000, temperature=0.7, do_sample=True)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("[/INST]")[-1].strip()
