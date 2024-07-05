import streamlit as st
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Load the tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
model = load_model('sentence_completion.h5')

# Function for text autocompletion
def autoCompletions(text, model, tokenizer, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')
    predicted_word_index = np.argmax(model.predict(sequence, verbose=0))
    predicted_word = ''
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break
    return text + ' ' + predicted_word

# Streamlit UI
st.title('Shakespeare Text Autocompletion')

input_text = st.text_input('Enter a starting phrase:', 'I love')

if st.button('Complete Text'):
    generated_text = input_text
    for _ in range(5):  # Change the number of words to generate
        generated_text = autoCompletions(generated_text, model, tokenizer, max_sequence_len=len(generated_text.split())+1)
    st.success(generated_text)

st.sidebar.header('About')
st.sidebar.markdown('This app uses an LSTM model to predict the next word based on a starting phrase from Shakespearean text.')
