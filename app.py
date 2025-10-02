from flask import Flask, render_template, request, jsonify
import os
from chatbot import ChatbotAssistant


app = Flask(__name__)
chatbot = ChatbotAssistant('intents.json')
chatbot.parse_intents()


def initialize_chatbot():
    model_exists = os.path.exists('rso_chatbot_model.pth') and os.path.exists('rso_chatbot_data.pkl')

    if model_exists:
        print("Loading existing model...")
        chatbot.load_model()
    else:
        print("Training new model...")
        chatbot.prepare_data()
        chatbot.train_model(batch_size=8, lr=0.001, epochs=200)
        chatbot.save_model()

    print("Chatbot ready!")


initialize_chatbot()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    response = chatbot.process_message(user_message)
    return jsonify({'response': response})


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, port=8000)
