import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing.text import Tokenizer
import regex as re
import pickle

# Load model architecture from JSON file
with open("lstm_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

with open("tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict next words
def predict_next_words(seed_text, model, tokenizer, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    return predicted

# Streamlit UI
st.title("Sentence Autocompletion App")
seed_text = st.text_input("Enter the seed text:")

if st.button("Complete Sentence"):
    max_sequence_len = 16  # Adjust as per your model's requirement
    predictions = predict_next_words(seed_text, loaded_model, tokenizer, max_sequence_len)
    
    # Assuming you want to display the top 5 predictions
    top_indices = predictions[0].argsort()[-5:][::-1]
    predicted_words = [tokenizer.index_word[idx] for idx in top_indices]
    print(predicted_words)
    
    st.write("Predicted next words:")
    for word in predicted_words:
        st.write(f"{seed_text} {word}")


