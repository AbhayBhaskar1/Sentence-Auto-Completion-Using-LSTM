import streamlit as st
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dropout

# Load the tokenizer and model
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Use custom_objects to handle Dropout layer
custom_objects = {'Dropout': Dropout}
model = load_model('sentence_completion.h5', custom_objects=custom_objects)

# Define the maximum sequence length used during training
max_sequence_len = 10  # Replace this with the actual value used during training

# Function for text autocompletion
def autoCompletions(text, model, tokenizer, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')
    predicted_word_index = np.argmax(model.predict(sequence, verbose=0))
    
    
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break
    
    # If the predicted word is None, handle the case
    if predicted_word is None:
        st.error("Error: Predicted word not found in tokenizer's vocabulary.")
        return text  # Return the original text to prevent further errors
    
    return text + ' ' + predicted_word

def generate_text(text, new_words, model, tokenizer, max_sequence_len):
    for _ in range(new_words):
        text = autoCompletions(text, model, tokenizer, max_sequence_len)
    return text

st.title('Shakespeare Text Autocompletion')

input_text = st.text_input('Enter a starting phrase:')

num_sentences = st.slider('Number of Words to generate:', 1, 10, 1)

if st.button('Complete Text'):
    generated_text = input_text.strip()
    if not generated_text:
        st.error("Please enter a valid starting phrase.")
    elif len(generated_text.split()) > 3:
        st.error("Input phrase must be 3 words or fewer.")
    else:
        generated_text = generate_text(generated_text, num_sentences, model, tokenizer, max_sequence_len)
        st.success(generated_text)

st.sidebar.header('About')
st.sidebar.markdown('This app uses an LSTM model to predict the next word based on a starting phrase from Shakespearean text.')
