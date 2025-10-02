import torch.nn as nn
import nltk
import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import pickle
import torch.nn.functional as F
import random


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.function_mappings = function_mappings or {}
        self.X = None
        self.y = None

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
        return words

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r', encoding='utf-8') as f:
                intents_data = json.load(f)

            all_words = []

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    all_words.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

            self.vocabulary = sorted(set(all_words))

            print(f"Loaded {len(self.intents)} intents with {len(self.vocabulary)} unique words")
        else:
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)
            intent_index = self.intents.index(document[1])
            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

        print(f"Training data prepared: {self.X.shape[0]} samples, {self.X.shape[1]} features")

    def train_model(self, batch_size=8, lr=0.001, epochs=100):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print("Starting training...")
        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                avg_loss = running_loss / len(loader)
                print(f"Epoch {epoch + 1}/{epochs}: Loss: {avg_loss:.4f}")

        print("Training completed!")

    def save_model(self, model_path='rso_chatbot_model.pth', data_path='rso_chatbot_data.pkl'):
        torch.save(self.model.state_dict(), model_path)
        data = {
            'vocabulary': self.vocabulary,
            'intents': self.intents,
            'intents_responses': self.intents_responses,
            'input_size': self.X.shape[1],
            'output_size': len(self.intents)
        }

        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved to {model_path}")
        print(f"Data saved to {data_path}")

    def load_model(self, model_path='rso_chatbot_model.pth', data_path='rso_chatbot_data.pkl'):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        self.vocabulary = data['vocabulary']
        self.intents = data['intents']
        self.intents_responses = data['intents_responses']
        self.model = ChatbotModel(data['input_size'], data['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        print("Model and data loaded successfully!")

    def get_confidence_score(self, predictions):
        probabilities = F.softmax(predictions, dim=1)
        max_prob = torch.max(probabilities).item()
        return max_prob

    def process_message(self, input_message, confidence_threshold=0.7):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(bag_tensor)

        confidence = self.get_confidence_score(predictions)
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if confidence < confidence_threshold:
            return ("I'm not sure about that. For specific RSO questions, please contact the RSO Specialist at "
                    "rso@berea.edu or visit the OSIE office in Alumni Building.")

        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return ("I understand your question, but I don't have a specific response. Please contact rso@berea.edu "
                    "for assistance.")

    def chat(self):
        print("=" * 60)
        print("RSO Chatbot Assistant - Berea College")
        print("Type 'quit' or 'exit' to end the conversation")
        print("=" * 60)

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nBot: Goodbye! For more RSO help, contact rso@berea.edu")
                break

            if not user_input:
                print("\nBot: Please type your question about RSOs.")
                continue

            response = self.process_message(user_input)
            print(f"\nBot: {response}")


def train_new_model():
    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=200)
    assistant.save_model()
    return assistant


def load_existing_model():
    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.load_model()
    return assistant


def main():
    model_exists = os.path.exists('rso_chatbot_model.pth') and os.path.exists('rso_chatbot_data.pkl')

    if model_exists:
        print("Existing model found. Loading...")
        assistant = load_existing_model()
    else:
        print("No existing model found. Training new model...")
        assistant = train_new_model()

    assistant.chat()


if __name__ == '__main__':
    main()
