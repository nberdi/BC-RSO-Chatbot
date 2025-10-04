# Berea College RSO Chatbot

A neural network-based chatbot that answers questions about Recognized Student Organizations (RSOs) at Berea College.

## Features

- 65 intent categories for RSO questions
- Neural network-based intent classification
- Web interface accessible locally or [online](https://bc-rso-chatbot.onrender.com)
- No API costs

## Tech Stack

- **Backend**: Flask, PyTorch
- **NLP**: NLTK
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Gunicorn

## Project Structure

```
BC-RSO-Chatbot/
├── README.md                 # Project documentation
├── app.py                    # Flask web server
├── chatbot.py                # Neural network and training logic
├── intents.json              # Training data (65 intents)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore file
├── templates/
│   └── index.html            # Chat interface
├── static/
│   ├── style.css             # Styling
│   ├── script.js             # Frontend logic
│   └── rso.png               # RSO logo
├── rso_chatbot_model.pth     # Trained model (generated)
└── rso_chatbot_data.pkl      # Model metadata (generated)
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nberdi/BC-RSO-Chatbot.git
cd BC-RSO-Chatbot
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Local Development

```bash
python app.py
```

Visit: `http://localhost:8000`

### Training

The chatbot trains automatically on first run. To retrain after updating `intents.json`:

```bash
rm rso_chatbot_model.pth rso_chatbot_data.pkl
python app.py
```

## Model Details

- **Architecture**: 3-layer feedforward neural network (input → 128 → 64 → output)
- **Training**: 200 epochs, Adam optimizer, CrossEntropyLoss
- **Features**: Bag of words encoding with lemmatization
- **Confidence threshold**: 70% (queries below threshold receive fallback response)

## Data Sources

Training data derived from official Berea College RSO handbooks:
- RSO Student Handbook
- RSO Advisor Handbook
- RSO Advisor Agreement

## Deployment

Configured for deployment on Render, Railway, or similar platforms. The app uses environment variable `PORT` and runs with Gunicorn in production.

## Limitations

- Responses are template-based, not generative
- Limited to information in training data
- Cannot access real-time data or external systems
- For complex or policy-specific questions, users are directed to contact OSIE

## Contact

RSO questions: rso@berea.edu