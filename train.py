import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import joblib
# Load IMDb dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    with_info=True,
    as_supervised=True
)
# Extract text and labels
train_sentences = []
train_labels = []
for text, label in tfds.as_numpy(train_data):
    train_sentences.append(text.decode('utf-8'))
    train_labels.append(label)
train_labels = np.array(train_labels)

tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=120, padding='post', truncating='post')

# Load GloVe
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create embedding matrix
embedding_dim = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((15000, embedding_dim))
for word, i in word_index.items():
    if i < 15000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Use in model
model = Sequential([
    Embedding(15000, embedding_dim, weights=[embedding_matrix], input_length=120, trainable=True),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

#Compile and train model with early stopping
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

model.fit(train_padded, train_labels, epochs=10, validation_split=0.2, callbacks=[early_stop])

#saving the model

model.save('model.keras') 

joblib.dump(tokenizer, 'tokenizer.pkl')