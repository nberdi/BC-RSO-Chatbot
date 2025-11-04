import os
import nltk
from flask import Flask, render_template, request, jsonify
from chatbot import ChatbotAssistant

app = Flask(__name__)

chatbot = None
COMMON_QUESTIONS = []


def download_nltk_data():
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]

    for path, name in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK data: {name}...")
            nltk.download(name, quiet=True)


def initialize_chatbot():
    global chatbot, COMMON_QUESTIONS

    try:
        chatbot = ChatbotAssistant('intents.json')
        chatbot.parse_intents()

        model_exists = (
                os.path.exists('rso_chatbot_model.pth') and
                os.path.exists('rso_chatbot_data.pkl')
        )

        if model_exists:
            print("Loading existing model...")
            chatbot.load_model()
        else:
            print("No existing model found. Training new model...")
            chatbot.prepare_data()
            chatbot.train_model(batch_size=8, lr=0.001, epochs=200)
            chatbot.save_model()

        COMMON_QUESTIONS = chatbot.get_common_questions()
        print(f"Loaded {len(COMMON_QUESTIONS)} common questions for suggestions")
        print("Chatbot ready!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure intents.json exists in the project directory")
        raise
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        raise


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Invalid request format',
                'status': 'error'
            }), 400

        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({
                'error': 'No message provided',
                'status': 'error'
            }), 400

        response = chatbot.process_message(user_message)

        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({
            'error': 'Something went wrong. Please try again.',
            'status': 'error'
        }), 500


@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'suggestions': []})

        query = data.get('query', '').lower().strip()

        # Require at least 2 characters for suggestions
        if len(query) < 2:
            return jsonify({'suggestions': []})

        # Filter questions that contain the query
        suggestions = [
            q for q in COMMON_QUESTIONS
            if query in q.lower()
        ]

        # Limit to 5 suggestions
        return jsonify({'suggestions': suggestions[:5]})

    except Exception as e:
        print(f"Error in /suggestions endpoint: {e}")
        return jsonify({'suggestions': []})


@app.route('/health')
def health():
    is_ready = chatbot is not None and chatbot.model is not None

    return jsonify({
        'status': 'healthy' if is_ready else 'initializing',
        'chatbot_loaded': is_ready,
        'suggestions_count': len(COMMON_QUESTIONS)
    })


def main():
    print("Starting RSO Chatbot Application...")

    download_nltk_data()

    initialize_chatbot()

    port = int(os.environ.get('PORT', 8000))

    print(f"Starting Flask server on port {port}...")
    app.run(
        debug=False,
        host='0.0.0.0',
        port=port
    )


if __name__ == '__main__':
    main()
