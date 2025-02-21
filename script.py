import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf
from tensorflow import keras


DATA_DIR = '/Users/drewcurley/Desktop/test/experiment/'  # <-- Set your directory here

def load_text_files(directory):
    texts = []
    for filename in sorted(os.listdir(directory)):
        if any(filename.startswith(str(i)) for i in range(41, 68)) and filename.endswith('.usfm'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.extend(file.readlines())
    return texts

texts = load_text_files(DATA_DIR)
print(f"Loaded {len(texts)} lines from text files 41 to 67.")


# Tokenize the text
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences
max_length = 50
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Prepare input for autoencoder (categorical labels)
x_train = np.array(padded_sequences)
y_train = keras.utils.to_categorical(x_train, num_classes=vocab_size)

from tensorflow.keras import layers

embedding_dim = 128
latent_dim = 64

def build_autoencoder():
    # Encoder
    encoder_input = layers.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(encoder_input)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)
    latent_vector = layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    x = layers.RepeatVector(max_length)(latent_vector)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    decoder_output = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)

    autoencoder = keras.Model(encoder_input, decoder_output)
    encoder = keras.Model(encoder_input, latent_vector)
    return autoencoder, encoder

autoencoder, encoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

def calculate_reconstruction_error(sentence):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict and calculate reconstruction error
    predicted_seq = autoencoder.predict(sequence)
    error = np.mean(np.abs(sequence - np.argmax(predicted_seq, axis=-1)))
    return error

# Example test
test_sentence_correct = "This is a correct sentence."
test_sentence_incorrect = "Ths is an incorect sentnce."

error_correct = calculate_reconstruction_error(test_sentence_correct)
error_incorrect = calculate_reconstruction_error(test_sentence_incorrect)

threshold = 0.05  # Adjust this threshold based on your dataset
print(f"Correct Sentence Error: {error_correct}")
print(f"Incorrect Sentence Error: {error_incorrect}")

if error_correct > threshold:
    print("Potential grammar or spelling error detected!")
else:
    print("Text appears correct.")
