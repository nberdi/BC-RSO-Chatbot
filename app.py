import nltk
from flask import Flask, render_template, request, jsonify
import os
from chatbot import ChatbotAssistant


try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


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
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided', 'status': 'error'}), 400

        response = chatbot.process_message(user_message)
        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Something went wrong', 'status': 'error'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)
