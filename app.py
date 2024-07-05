import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
data = pd.read_csv('Shakespeare.csv')
print(data.head())

# Getting text from the data
text = data['PlayerLine'].tolist()

# Text Cleaning
def clean_text(text):
    # Removing special characters and digits
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = re.sub('\d+', '', text)
    # Converting text to lower case
    text = text.lower()
    return text

texts = [clean_text(t) for t in text]

# Let's take the first 10000 cleaned lines for model training
texts = texts[:10000]

# Using TensorFlow tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Generating text sequences
text_sequences = tokenizer.texts_to_sequences(texts)

# Padding the sequences
max_sequence_len = max([len(x) for x in text_sequences])
text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_len, padding='pre')

print('Maximum Sequence Length -->>', max_sequence_len)
print('Text Sequence -->>\n', text_sequences[0])
print('Text Sequence Shape -->>', text_sequences.shape)

# Getting X and y from the data
X, y = text_sequences[:, :-1], text_sequences[:, -1]
print('First Input :', X[0])
print('First Target :', y[0])

word_index = tokenizer.word_index

# Using one-hot encoding on y
total_words = len(word_index) + 1
print('Total Number of Words:', total_words)

y = to_categorical(y, num_classes=total_words)

# Printing X and y shapes
print('Input Shape :', X.shape)
print('Target Shape :', y.shape)

# Building the model
model = Sequential(name="LSTM_Model")
model.add(Embedding(total_words, max_sequence_len-1, input_length=max_sequence_len-1))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation='softmax'))

# Printing model summary
print(model.summary())

# Compiling the model
model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

# Training the LSTM model
history = model.fit(X, y, epochs=50, verbose=1)

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

# Generate text with specified number of new words
def generate_text(text, new_words, model, tokenizer, max_sequence_len):
    for _ in range(new_words):
        text = autoCompletions(text, model, tokenizer, max_sequence_len)
    return text

# Example of generated text
generated_text = generate_text('I love', 5, model, tokenizer, max_sequence_len)
print(generated_text)

# Saving the model
model.save('sentence_completion.h5')

# Saving the tokenizer
filename = 'tokenizer.pkl'
pickle.dump(tokenizer, open(filename, 'wb'))
