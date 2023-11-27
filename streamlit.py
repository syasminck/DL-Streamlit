import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SimpleRNN, Dropout, Conv2D, MaxPooling2D, Flatten
#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import cv2
import os
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import imdb

# Load IMDb dataset for sentiment analysis
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=5000)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
max_length = 500
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)
# Tokenization and Padding
tokeniser = tf.keras.preprocessing.text.Tokenizer()
X_train_texts = [str(sentence) for sentence in X_train]
# Then apply the tokenizer
tokeniser.fit_on_texts(X_train_texts)
encoded = tokeniser.texts_to_sequences(X_train_texts)
max_length = 500

padded = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=max_length, padding='post')
vocab_size = len(tokeniser.word_index) + 1

# Load Tumor dataset for tumor detection
image_dir = r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ALGORITHMS\CNN\tumor_detection\tumordata'
no_tumor_images = os.listdir(image_dir + '/no')
yes_tumor_images = os.listdir(image_dir + '/yes')

dataset = []
label = []
img_size = (128, 128)

for i, image_name in tqdm(enumerate(no_tumor_images), desc="No Tumor"):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_dir + '/no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(img_size)
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in tqdm(enumerate(yes_tumor_images), desc="Tumor"):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_dir + '/yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize(img_size)
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

# Normalize the dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Model 1: Perceptron
perceptron_model = Perceptron(max_iter=5)
perceptron_model.fit(X_train, Y_train)

# Model 2: Backpropagation
backprop_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5, random_state=42)
backprop_model.fit(X_train, Y_train)

# Model 3: DNN
dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, Y_train, epochs=5, batch_size=32)

# Model 4: RNN
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=24, input_length=max_length),
    SimpleRNN(24, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, Y_train, epochs=5, batch_size=32)

# Model 5: LSTM

lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=24, input_length=max_length),
    LSTM(24, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train, Y_train, epochs=5, batch_size=32)

# Placeholder code for training CNN model for tumor detection (Replace with actual training code)
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(x_train, y_train, epochs=5, batch_size=32)

# Streamlit app
def main():
    st.title("Sentiment Analysis and Tumor Detection App")
    task_choice = st.radio("Choose a task:", ["Tumor Detection", "Sentiment Analysis"])

    if task_choice == "Tumor Detection":
        st.subheader("Tumor Detection")
        image_upload = st.file_uploader("Upload an image for tumor detection", type=["jpg", "jpeg", "png"])

        if image_upload:
            # Load the CNN model for tumor detection
            tumor_model = tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')  # Replace with your actual model path

            # Preprocess the uploaded image
            image = Image.open(image_upload).convert("RGB")
            image = image.resize((128, 128))
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)

            # Make prediction
            prediction = tumor_model.predict(image_array)
            if prediction > 0.5:
                st.success("Tumor Detected in the uploaded image!")
            else:
                st.success("No Tumor Detected in the uploaded image!")

    elif task_choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        model_choice = st.selectbox("Choose a Sentiment Analysis Model:", ["Perceptron", "Backpropagation", "DNN", "RNN", "LSTM"])

        # Load the selected sentiment analysis model
        sentiment_model = load_sentiment_model(model_choice)

        # Input paragraph for sentiment analysis
        input_paragraph = st.text_area("Enter the paragraph for sentiment analysis:")

        if st.button("Analyze Sentiment"):
            if input_paragraph:
                # Tokenize and pad the input paragraph
                input_sequence = [word_index[word] if word in word_index else 0 for word in input_paragraph.split()]
                input_sequence = pad_sequences([input_sequence], maxlen=max_length)
                
                # Make prediction
                prediction = sentiment_model.predict(input_sequence)
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                
                st.success(f"The sentiment of the input paragraph is: {sentiment}")

def load_sentiment_model(model_name):
    # Load and return the selected sentiment analysis model (Replace with actual model paths)
    if model_name == "Perceptron":
        return tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')
    elif model_name == "Backpropagation":
        return tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')
    elif model_name == "DNN":
        return tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')
    elif model_name == "RNN":
        return tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')
    elif model_name == "LSTM":
        return tf.keras.models.load_model(r'C:\Users\USER\OneDrive\Desktop\deep-learning\DL-ASSN\streamlit.py')

if __name__ == "__main__":
    main()
