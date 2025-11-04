import os
import json
import pickle
import random
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import nltk
from spellchecker import SpellChecker


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

        self.spell = SpellChecker()

        self._download_nltk_data()

    def _download_nltk_data(self):
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

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
        return words

    def correct_spelling(self, text):
        words = text.split()
        corrected_words = []

        for word in words:
            # Keep very short words as-is
            if len(word) <= 2 or not word.isalpha():
                corrected_words.append(word)
            else:
                # Correct the word
                corrected = self.spell.correction(word.lower())
                corrected_words.append(corrected if corrected else word)

        return ' '.join(corrected_words)

    @staticmethod
    def is_valid_input(text):
        text_no_spaces = text.replace(" ", "").strip()

        # Too short
        if len(text_no_spaces) < 2:
            return False

        # Check if mostly alphabetic
        alpha_chars = sum(1 for char in text_no_spaces if char.isalpha())
        if alpha_chars == 0:
            return False

        # At least 70% should be letters
        alpha_ratio = alpha_chars / len(text_no_spaces)
        if alpha_ratio < 0.7:
            return False

        # Check vowel ratio
        vowels = sum(1 for char in text_no_spaces.lower() if char in 'aeiou')
        consonants = sum(
            1 for char in text_no_spaces.lower()
            if char.isalpha() and char not in 'aeiou'
        )

        if consonants > 0:
            vowel_ratio = vowels / (vowels + consonants)
            if vowel_ratio < 0.20:
                return False

        # Check for repeated characters
        if re.search(r'(.)\1{4,}', text_no_spaces):
            return False

        # Check for excessive consonant clusters
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text_no_spaces.lower()):
            return False

        return True

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Intents file not found: {self.intents_path}")

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

    def train_model(self, batch_size=8, lr=0.001, epochs=200):
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

    def process_message(self, input_message, confidence_threshold=0.85):
        # Validate input
        if not self.is_valid_input(input_message):
            return (
                "I didn't quite understand that. Please contact the RSO office "
                "at rso@berea.edu for assistance with your question."
            )

        # Correct spelling
        corrected_message = self.correct_spelling(input_message)

        # Tokenize
        words = self.tokenize_and_lemmatize(corrected_message)

        if not words:
            return (
                "I didn't quite understand that. Please contact the RSO office "
                "at rso@berea.edu for assistance with your question."
            )

        # Check vocabulary match
        matched_words = [w for w in words if w in self.vocabulary]

        if len(matched_words) / len(words) < 0.3:
            return (
                "I'm not sure I can help with that. For RSO-related questions, "
                "please contact rso@berea.edu or visit the OSIE office in Alumni Building."
            )

        # Get prediction
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        confidence = self.get_confidence_score(predictions)
        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        # Check confidence
        if confidence < confidence_threshold:
            return (
                "I'm not sure about that specific question. Please contact the RSO Specialist "
                "at rso@berea.edu or visit the OSIE office in Alumni Building (ext. 3290) for assistance."
            )

        # Execute mapped functions if any
        if self.function_mappings and predicted_intent in self.function_mappings:
            self.function_mappings[predicted_intent]()

        # Return response
        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return ("I understand your question, but I don't have a specific response. Please contact rso@berea.edu "
                    "for assistance.")

    def get_common_questions(self):
        questions = []

        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            for pattern in intent['patterns']:
                question = self._pattern_to_question(pattern)
                if question:
                    questions.append(question)

        return questions

    @staticmethod
    def _pattern_to_question(pattern):
        pattern = pattern.strip()

        # Already a question
        if pattern.endswith('?'):
            return pattern

        # Starts with question word
        question_starters = ['how', 'what', 'when', 'where', 'who', 'why', 'can', 'is', 'are', 'do', 'does']
        first_word = pattern.lower().split()[0] if pattern.split() else ''

        if first_word in question_starters:
            return pattern.capitalize() + '?'

        # Skip short keyword patterns
        if len(pattern.split()) < 3:
            return None

        return pattern.capitalize() + '?'

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
    model_exists = (
            os.path.exists('rso_chatbot_model.pth') and
            os.path.exists('rso_chatbot_data.pkl')
    )

    if model_exists:
        print("Existing model found. Loading...")
        assistant = load_existing_model()
    else:
        print("No existing model found. Training new model...")
        assistant = train_new_model()

    assistant.chat()


if __name__ == '__main__':
    main()
