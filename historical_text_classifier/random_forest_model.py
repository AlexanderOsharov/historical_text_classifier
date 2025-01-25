import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import chardet
from os.path import join, dirname, abspath
import joblib

class RandomForestTextClassifier:
    def __init__(self, dataset_path=None, model_path=None):
        self.dataset_path = dataset_path or join(dirname(abspath(__file__)), 'data', 'dataset.json')
        self.model_path = model_path or 'random_forest_model.pkl'
        self.model = None
        self.vectorizer = None
        self.stop_words = set(['акт', 'лист', 'приказ', 'распоряжение', '№'])

    def preprocess_text(self, text):
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9.,!?\s]", "", text)  # Удаление лишних символов
        text = text.lower()  # Приведение к нижнему регистру
        words = text.split()
        words = [word for word in words if word not in self.stop_words]  # Удаление стоп-слов
        return ' '.join(words)

    def load_data(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        labels = [1 if item["label"] == "historical_background" else 0 for item in data]
        return texts, labels

    def train(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.save_model()

    def save_model(self):
        joblib.dump((self.model, self.vectorizer), self.model_path)

    def load_model(self):
        self.model, self.vectorizer = joblib.load(self.model_path)

    def predict(self, text):
        if not self.model or not self.vectorizer:
            self.load_model()
        preprocessed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([preprocessed_text])
        return self.model.predict(text_vector)[0]

    def predict_proba(self, text):
        if not self.model or not self.vectorizer:
            self.load_model()
        preprocessed_text = self.preprocess_text(text)
        text_vector = self.vectorizer.transform([preprocessed_text])
        return self.model.predict_proba(text_vector)[0]

    def evaluate(self):
        texts, labels = self.load_data()
        texts = [self.preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(texts)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        print("Accuracy:", accuracy_score(y_test, y_pred))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Twaddle', 'Historical Background'], yticklabels=['Twaddle', 'Historical Background'])
        plt.xlabel('Предсказанные значения')
        plt.ylabel('Фактические значения')
        plt.title('Матрица ошибок')
        plt.show()

        # ROC-кривая
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend(loc="lower right")
        plt.show()

    def extract_valuable_passages(self, input_text, threshold=0.5, min_length=10):
        paragraphs = input_text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip().split()) >= min_length]
        paragraphs = [p for p in paragraphs if not any(kw in p.lower() for kw in self.stop_words)]
        preprocessed_paragraphs = [self.preprocess_text(p) for p in paragraphs]
        paragraph_vectors = self.vectorizer.transform(preprocessed_paragraphs)
        probabilities = self.model.predict_proba(paragraph_vectors)[:, 1]
        results = [(para, prob) for para, prob in zip(paragraphs, probabilities) if prob >= threshold]
        return results

    def fetch_data_from_wikipedia(self, query, num_results=5):
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("query", {}).get("search", [])
            return [item["snippet"] for item in results][:num_results]
        else:
            return []