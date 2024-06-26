import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import cm
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load data
df = pd.read_csv('data_data_berita_cleaned_preprocess.csv')

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

# Function to preprocess text


def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and not word.isdigit()]
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# Function to generate word cloud


def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='black').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Function to generate bar chart


def generate_barchart(word_counts):
    words, counts = zip(*word_counts)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, counts, color=cm.viridis(range(len(words))))
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Words')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Main function


def main():
    st.sidebar.title('Navigation')
    app_mode = st.sidebar.selectbox(
        'Choose the App Mode', ['Home', 'News Category Prediction', 'EDA'])

    if app_mode == 'Home':
        st.title('Welcome to News Analysis Dashboard')
        st.write(
            'This dashboard allows you to predict news categories and visualize data.')
        st.write('Please select an option from the sidebar.')

    elif app_mode == 'News Category Prediction':
        st.title('News Article Category Prediction')
        st.write('Enter your news article below to predict its category.')

        user_input = st.text_area("Input News Article", "")

        if st.button('Predict'):
            preprocessed_input = preprocess_text(user_input)
            padded_input = pad_sequences(tokenizer.texts_to_sequences(
                [preprocessed_input]), padding='pre', truncating='pre', maxlen=maxlen)
            prediction = model.predict(padded_input)
            categories = ['Kesehatan', 'Keuangan', 'Kuliner',
                          'Olahraga', 'Otomotif', 'Pariwisata', 'Pendidikan']
            predicted_category = categories[np.argmax(prediction)]
            st.success(f'Predicted Category: {predicted_category}')

    elif app_mode == 'EDA':
        st.title('Dashboard Word Cloud and Bar Chart')
        st.write(
            'Explore word cloud and bar chart visualization for news categories.')

        categories = df['kategori'].unique()
        selected_category = st.selectbox('Choose a Category', categories)

        category_df = df[df['kategori'] == selected_category]
        combined_text = ' '.join(category_df['isi'])

        st.subheader(f'Word Cloud for Category: {selected_category}')
        generate_wordcloud(combined_text)

        word_counts = Counter(combined_text.split())
        common_words = word_counts.most_common(10)

        st.subheader(f'Top 10 Words in Category: {selected_category}')
        generate_barchart(common_words)


if __name__ == '__main__':
    main()
