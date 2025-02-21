from flask import Flask, request, jsonify
from main import load_trained_model
import torch

def top_k_prediction(model, tokenizer, text, k=5):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=len(inputs[0]) + 5, num_return_sequences=k, do_sample=True, top_k=50)
    
    predictions = []
    for output in outputs:
        prediction = tokenizer.decode(output, skip_special_tokens=True)
        predictions.append(prediction)
    
    return predictions

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data['text']
    k = data.get('k', 5)
    
    model, tokenizer = load_trained_model()
    
    predictions = top_k_prediction(model, tokenizer, user_input, k=k)
    
    return jsonify({'predictions': predictions})

if __name__ == "__main__":
    app.run(debug=True)
