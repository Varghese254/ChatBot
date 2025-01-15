from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Chat history
chat_history_ids = None

@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html page

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    data = request.get_json()
    prompt = data.get("prompt", "")

    try:
        # Encode user input and add to chat history
        input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        if chat_history_ids is not None:
            chat_history_ids = torch.cat([chat_history_ids, input_ids], dim=-1)
        else:
            chat_history_ids = input_ids

        # Generate a response
        response_ids = model.generate(
            chat_history_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(response_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
