import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk


nltk.download('punkt')
nltk.download('stopwords')


# Load stemmer from pickle
with open('stemmer.pickle', 'rb') as f:
    stemmer = pickle.load(f)

# Load stopwords from pickle
with open('stopwords.pickle', 'rb') as f:
    stop_words = pickle.load(f)

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load model
model = load_model('model_berita_classification.h5')

# Load maxlen
with open('maxlen.txt', 'r') as f:
    maxlen = int(f.read())

# Preprocess function


def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Convert words to lowercase
    words = [word.lower() for word in words]

    # Remove punctuation and numbers
    words = [word for word in words if word.isalnum() and not word.isdigit()]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Perform stemming
    stemmed_words = [stemmer.stem(word) for word in words]

    # Join the words back into a single string
    return ' '.join(stemmed_words)

# Streamlit app


def main():
    st.title('News Article Category Prediction')
    st.write('Enter your news article below to predict its category.')

    # Input text area for user input
    user_input = st.text_area("Input News Article", "")

    if st.button('Predict'):
        # Preprocess user input
        preprocessed_input = preprocess_text(user_input)
        padded_input = pad_sequences(tokenizer.texts_to_sequences(
            [preprocessed_input]), padding='pre', truncating='pre', maxlen=maxlen)

        # Predict category
        prediction = model.predict(padded_input)
        categories = ['Kesehatan', 'Keuangan', 'Kuliner',
                      'Olahraga', 'Otomotif', 'Pariwisata', 'Pendidikan']
        predicted_category = categories[np.argmax(prediction)]

        # Display prediction
        st.success(f'Predicted Category: {predicted_category}')


if __name__ == '__main__':
    main()
