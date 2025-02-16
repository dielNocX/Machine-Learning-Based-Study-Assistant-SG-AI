import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random as rd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

# Download dari NLTK
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

normalization_dict = {
    "yg": "yang",
    "gak": "tidak",
    "bgt": "banget",
    "bener": "benar",
    "tp": "tapi",
    "aja": "saja",
    "blm": "belum",
    "krn": "karena"
}

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def fetch_data():
    # fetch dataset for model training
    df = pd.read_csv('./question.csv')
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

def normalize_text(text):
    words = text.split()
    words = [normalization_dict[word] if word in normalization_dict else word for word in words]
    return " ".join(words)


def preprocess_text(text):
    text = text.lower()  # Ubah ke huruf kecil lebih awal
    text = normalize_text(text)  # Normalisasi kata tidak baku setelah lowercase
    words = word_tokenize(text)  # Tokenisasi langsung dari hasil normalisasi
    ignored_words = {'belajar', 'aku', 'gue', 'gw','saya', 'banget'}
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and word not in ignored_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    
    return " ".join(stemmed_words)


# Fetch dan preprocess data teks
df = fetch_data()
x_train, x_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, random_state=10)

# data vektor TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.8, max_features=10000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train).toarray()
x_test_tfidf = tfidf_vectorizer.transform(x_test).toarray()

# pemilihan dan pelatihan model pada data teks
nb_model = GaussianNB()
nb_model.fit(x_train_tfidf, y_train)

def model_accuracy(model):
    y_pred = model.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    model_name = model.__class__.__name__
    return accuracy, report, model_name, y_test, y_pred

def display_evaluation(accuracy, report, model_name):
    st.write(f'Model: \t\t{model_name}')
    st.write(f'Accuracy: \t\t{accuracy * 100:.1f}%')
    st.write(f'Classification Report: \n{report}')


def evaluate():
    accuracy, report, model_name, y_test, y_pred = model_accuracy(nb_model)
    display_evaluation(accuracy, report, model_name)
    
def predict_sentiment(text):
    # preproses teks dan prediksi sentimen
    processed_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([processed_text]).toarray()
    prediction = nb_model.predict(vectorized_text)
    return prediction[0]  #mengembelikan prediksi sentimen

def random_response(sentiment):
    # fungsi yang memilih respons acak berdasarkan sentimen
    response = pd.read_csv('./response.csv')
    response_sentiment = response[response['sentiment'] == sentiment]
    if response_sentiment.empty:
        return "Maaf, bot belum tersedia."
    random_response = rd.choice(response_sentiment['response'].tolist())
    return random_response

def display_chat(role, chat):
    with st.chat_message(role):
        st.markdown(chat)

def random_intro():
    response = pd.read_csv('./intro.csv')
    return rd.choice(response['text'].tolist())

def chatbot():
    prompt = st.chat_input("Tanya seputar tips belajar (cth: saya butuh metode belajar yang bagus)")

    if prompt and (prompt == "Tanya seputar tips belajar (cth: saya butuh metode belajar yang bagus)" or any(x in prompt.lower() for x in ["halo", "hai","hi"])):
        user, bot= ("user", prompt), ("bot", random_intro())
        display_chat(*user)
        display_chat(*bot)
    elif prompt:
        user, bot = ("user", prompt), ("bot", random_response(predict_sentiment(prompt)))
        display_chat(*user)
        display_chat(*bot)
        # evaluate()

chatbot()
