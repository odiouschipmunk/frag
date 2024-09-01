from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
model_path = r"C:\Users\default.DESKTOP-7FKFEEG\project\model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

def query_model(input_text, max_new_tokens=5000):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    
    # Generate predictions with max_new_tokens
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # Decode the predictions
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return decoded_output

# Example usage
input_text = "what are all of the notes in dior savauge? I am talking about the fragrance."
output_text = query_model(input_text)
print("Output:", output_text)