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


data = pd.read_csv('Shakespeare.csv')
print(data.head())


text = data['PlayerLine'].tolist()


def clean_text(text):
  
    text = re.sub('[^a-zA-Z\s]', '', text)
    text = re.sub('\d+', '', text)
 
    text = text.lower()
    return text

texts = [clean_text(t) for t in text]


texts = texts[:10000]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)


text_sequences = tokenizer.texts_to_sequences(texts)


max_sequence_len = max([len(x) for x in text_sequences])
text_sequences = pad_sequences(text_sequences, maxlen=max_sequence_len, padding='pre')

print('Maximum Sequence Length -->>', max_sequence_len)
print('Text Sequence -->>\n', text_sequences[0])
print('Text Sequence Shape -->>', text_sequences.shape)


X, y = text_sequences[:, :-1], text_sequences[:, -1]
print('First Input :', X[0])
print('First Target :', y[0])

word_index = tokenizer.word_index


total_words = len(word_index) + 1
print('Total Number of Words:', total_words)

y = to_categorical(y, num_classes=total_words)


print('Input Shape :', X.shape)
print('Target Shape :', y.shape)

model = Sequential(name="LSTM_Model")
model.add(Embedding(total_words, max_sequence_len-1, input_length=max_sequence_len-1))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(total_words, activation='softmax'))


print(model.summary())


model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)


history = model.fit(X, y, epochs=50, verbose=1)


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


def generate_text(text, new_words, model, tokenizer, max_sequence_len):
    for _ in range(new_words):
        text = autoCompletions(text, model, tokenizer, max_sequence_len)
    return text


generated_text = generate_text('I love', 5, model, tokenizer, max_sequence_len)
print(generated_text)


model.save('sentence_completion.h5')


filename = 'tokenizer.pkl'
pickle.dump(tokenizer, open(filename, 'wb'))
